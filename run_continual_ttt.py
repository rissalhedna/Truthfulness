#!/usr/bin/env python3
"""
Continual Test-Time Training with Discriminative Calibration (SECL).

Adapts LLM confidence via lightweight LoRA updates using the
generation-discrimination gap as label-free self-supervision.

Usage:
    python run_continual_ttt.py --mode sequential --questions_per_domain 500
"""

import os
import argparse

# Parse args first to get GPU before any CUDA imports
parser = argparse.ArgumentParser(description='Continual TTT with Discriminative Calibration')
parser.add_argument('--gpu', type=int, default=3, help='GPU device ID')
parser.add_argument('--mode', type=str, default='sequential',
                    choices=['sequential', 'sequential_gen',
                             'sequential_hard', 'interleaved',
                             'return_gsm8k', 'return_mmlu',
                             'truthful_first', 'reversed', 'reversed_gen',
                             'mmlu_first', 'arc_first', 'custom'],
                    help='Benchmark mode')
parser.add_argument('--domains', type=str, nargs='+', default=None,
                    help='Custom domain order (use with --mode custom)')
parser.add_argument('--questions_per_domain', type=int, default=500, help='Questions per domain')
parser.add_argument('--n_neighbors', type=int, default=0, help='Number of neighborhood questions (0 = train on input question only)')
parser.add_argument('--n_epochs', type=int, default=3, help='Training epochs per question')
parser.add_argument('--k_distractors', type=int, default=4, help='Number of distractors for P(True) normalization')
parser.add_argument('--use_raw_ptrue', action='store_true', help='Use raw P(True) without normalization for target bin')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
parser.add_argument('--lora_alpha', type=int, default=16, help='LoRA alpha (scaling factor)')
parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Main model')
parser.add_argument('--neighbor_model', type=str, default='meta-llama/Llama-3.2-3B-Instruct', help='Neighbor generator model')
parser.add_argument('--no_wandb', action='store_true', help='Disable W&B logging')
parser.add_argument('--wandb_project', type=str, default='continual-ttt', help='W&B project name')
parser.add_argument('--run_name', type=str, default=None, help='W&B run name')
parser.add_argument('--accumulation', action='store_true', default=False, help='Accumulate LoRA weights')
parser.add_argument('--no_accumulation', action='store_true', default=False, help='Reset LoRA weights each question')
parser.add_argument('--baseline_only', action='store_true', help='Run baseline only (no TTT)')
parser.add_argument('--use_p_know', action='store_true', help='Use P(Know) instead of P(True) for discrimination')
parser.add_argument('--use_fused_confidence', action='store_true',
                    help='Use fused confidence (p_true_norm * mean P(Know across neighbors)) to set target bin')
parser.add_argument('--baseline_use_ptrue_norm', action='store_true',
                    help='In baseline_only mode, use P(True)_norm mapped to bin as the reported confidence')
parser.add_argument('--temperature_scaling', action='store_true',
                    help='Apply temperature scaling baseline (uses first 100 questions per domain as validation set)')
parser.add_argument('--norm_temperature', type=float, default=0.7,
                    help='Temperature (<1 sharpen) when normalizing P(True) over answer+distractors')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_bins', type=int, default=10, help='Number of confidence bins (rolling quantile edges)')
parser.add_argument('--bin_window', type=int, default=500, help='Rolling window size for bin quantiles')
parser.add_argument('--bin_min_count', type=int, default=100, help='Minimum samples before using quantile edges')
parser.add_argument('--fixed_bins', action='store_true', help='Use fixed uniform bin edges instead of adaptive quantile edges')
parser.add_argument('--neighbors_only', action='store_true', help='Train on generated neighbors only, exclude the input question from training')
parser.add_argument('--reuse_mcq_options', action='store_true',
                    help='For MCQ datasets, reuse original question MCQ options to normalize P(True) for neighbors '
                         '(default: neighbors generate their own distractors)')
# PH-gated TTT arguments
parser.add_argument('--use_ph_gate', action='store_true', help='Enable PH-gated TTT (only train on detected drift)')
parser.add_argument('--ph_delta', type=float, default=0.05, help='PH tolerance parameter')
parser.add_argument('--ph_threshold', type=float, default=4.0, help='PH trigger threshold')
parser.add_argument('--ema_alpha', type=float, default=0.05, help='EMA smoothing factor (lower=smoother)')
parser.add_argument('--ttt_burst', type=int, default=20, help='Number of TTT rounds per trigger')
parser.add_argument('--ttt_warmup', type=int, default=50, help='Force TTT for first N questions (warmup)')
parser.add_argument('--ph_reset_on_trigger', action='store_true', help='Zero LoRA weights when PH fires a new trigger (not during burst)')
parser.add_argument('--ph_cooldown', type=int, default=0, help='After burst ends, suppress PH triggers for N questions')
parser.add_argument('--use_bin_gate', action='store_true', help='Train only when |verbalized_bin - P(True)_bin| > threshold')
parser.add_argument('--bin_gate_threshold', type=int, default=2, help='Bin gap threshold (default 2)')
parser.add_argument('--use_base_target', action='store_true', help='Compute P(True) target using base model (no adapter) instead of adapted model')
parser.add_argument('--overconfident_only', action='store_true', help='With bin_gate: only train when model is overconfident (verbalized > P(True))')
parser.add_argument('--report_ptrue', action='store_true', help='After TTT, report P(True) norm as confidence instead of verbalized confidence')
parser.add_argument('--use_mse_loss', action='store_true', help='Use MSE loss on expected confidence from logits instead of CE on bin digit')
parser.add_argument('--use_directional_target', action='store_true', help='With MSE loss, move confidence partially toward P(True) target instead of exact matching')
parser.add_argument('--directional_alpha', type=float, default=0.3, help='Step size toward P(True) when using directional target (0-1)')
parser.add_argument('--directional_clip', type=float, default=0.15, help='Max absolute directional correction per sample when using directional target')
parser.add_argument('--use_ranking_loss', action='store_true', help='Use pairwise ranking loss against buffer of past questions')
parser.add_argument('--ranking_buffer_size', type=int, default=16, help='Number of past questions in ranking buffer')
parser.add_argument('--use_soft_confidence', action='store_true', help='Report soft expected confidence from digit logits instead of hard argmax')
parser.add_argument('--kl_reg_beta', type=float, default=0.0, help='KL divergence regularization weight against base model (0=disabled)')
# Layer targeting arguments
parser.add_argument('--layer_start', type=int, default=24, help='First layer to apply LoRA (default: 24 for late layers)')
parser.add_argument('--layer_end', type=int, default=32, help='Last layer (exclusive) to apply LoRA (default: 32)')
parser.add_argument('--lora_last_n_layers', type=int, default=0, help='If >0, auto-target last N layers (overrides layer_start/layer_end)')
parser.add_argument('--lora_target_keys', type=str, default='q_proj,v_proj', help='Comma-separated attention module suffixes to target (e.g. q_proj,v_proj or qkv_proj)')
parser.add_argument('--trust_remote_code', action='store_true', help='Pass trust_remote_code=True to model/tokenizer loading (default: False)')
parser.add_argument('--use_sc_target', action='store_true', help='Use Self-Consistency agreement as TTT target instead of P(True) Norm')
parser.add_argument('--sc_n_samples', type=int, default=10, help='Number of temperature samples for SC target')
parser.add_argument('--sc_temperature', type=float, default=0.7, help='Sampling temperature for SC target')
args = parser.parse_args()

# Handle accumulation flags
if args.no_accumulation:
    args.accumulation = False
elif not args.accumulation and not args.no_accumulation:
    args.accumulation = True

# SET GPU BEFORE ANY IMPORTS
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn.functional as F
import numpy as np
import json
import re
import random
import time
from datetime import datetime
from collections import defaultdict, deque
from contextlib import nullcontext
from functools import partial
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from sklearn.metrics import roc_auc_score
from utils import (
    create_qa_prompt as _create_qa_prompt,
    extract_answer,
    extract_confidence as _extract_confidence,
    answers_match,
    load_qa_dataset,
    normalize_answer,
)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)

DEVICE = "cuda:0"

# =============================================================================
# CONFIGURATION
# =============================================================================

MODEL_NAME = args.model_name
NEIGHBOR_MODEL_NAME = args.neighbor_model

DOMAIN_CONFIG = {
    'gsm8k': {'dataset_mode': 'gsm8k', 'dataset_type': 'gsm8k', 'name': 'GSM8K'},
    'mmlu': {'dataset_mode': 'mmlu', 'dataset_type': 'mmlu', 'name': 'MMLU'},
    'arc': {'dataset_mode': 'arc', 'dataset_type': 'arc', 'name': 'ARC'},
    'truthfulqa': {'dataset_mode': 'truthfulqa_mc', 'dataset_type': 'truthfulqa_mc', 'name': 'TruthfulQA-MC'},
    'truthfulqa_gen': {'dataset_mode': 'truthfulqa', 'dataset_type': 'truthfulqa', 'name': 'TruthfulQA (gen)'},
    'hle': {'dataset_mode': 'hle', 'dataset_type': 'hle', 'name': 'HLE'},
}

BENCHMARK_MODES = {
    'sequential': ['gsm8k', 'mmlu', 'arc', 'truthfulqa'],
    'reversed': ['truthfulqa', 'arc', 'mmlu', 'gsm8k'],
    'sequential_gen': ['gsm8k', 'mmlu', 'arc', 'truthfulqa_gen'],
    'reversed_gen': ['truthfulqa_gen', 'arc', 'mmlu', 'gsm8k'],
    'sequential_hard': ['gsm8k', 'mmlu', 'arc', 'truthfulqa', 'hle'],
    'interleaved': ['gsm8k', 'mmlu', 'arc', 'truthfulqa'],
    'return_gsm8k': ['gsm8k', 'mmlu', 'gsm8k'],
    'return_mmlu': ['mmlu', 'arc', 'mmlu'],
    'truthful_first': ['truthfulqa', 'gsm8k', 'mmlu', 'arc'],
    'mmlu_first': ['mmlu', 'arc', 'truthfulqa', 'gsm8k'],
    'arc_first': ['arc', 'truthfulqa', 'gsm8k', 'mmlu'],
}

LAYER_START = args.layer_start
LAYER_END = args.layer_end

NUM_BINS = args.num_bins
CONF_TO_PROB = {i: (i + 0.5) / NUM_BINS for i in range(NUM_BINS)}

create_qa_prompt = partial(_create_qa_prompt, num_bins=NUM_BINS)
extract_confidence = partial(_extract_confidence, num_bins=NUM_BINS)

print("=" * 70)
print("CONTINUAL TTT WITH DISCRIMINATIVE CALIBRATION")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# =============================================================================
# LOAD MODELS
# =============================================================================

print(f"Loading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=args.trust_remote_code)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=args.trust_remote_code,
    low_cpu_mem_usage=True
).to(DEVICE)
base_model.eval()

# Clamp requested LoRA layer range to actual model depth.
num_hidden_layers = getattr(base_model.config, "num_hidden_layers", None)
if isinstance(num_hidden_layers, int) and num_hidden_layers > 0:
    if args.lora_last_n_layers > 0:
        LAYER_START = max(0, num_hidden_layers - args.lora_last_n_layers)
        LAYER_END = num_hidden_layers
        print(f"Auto-targeting last {args.lora_last_n_layers} layers: [{LAYER_START}, {LAYER_END})")
    orig_start, orig_end = LAYER_START, LAYER_END
    LAYER_START = max(0, min(LAYER_START, num_hidden_layers - 1))
    LAYER_END = max(LAYER_START + 1, min(LAYER_END, num_hidden_layers))
    if (orig_start, orig_end) != (LAYER_START, LAYER_END):
        print(
            f"Adjusted layer range from [{orig_start}, {orig_end}) "
            f"to [{LAYER_START}, {LAYER_END}) for model depth={num_hidden_layers}"
        )

print(f"Base model loaded to: {next(base_model.parameters()).device}")
print(f"Model layers targeted: {LAYER_START}-{LAYER_END-1}")

if NEIGHBOR_MODEL_NAME == MODEL_NAME:
    print(f"Neighbor model == base model ({NEIGHBOR_MODEL_NAME}); reusing base model/tokenizer for neighbor generation.")
    neighbor_tokenizer = tokenizer
    neighbor_model = base_model
else:
    print(f"Loading {NEIGHBOR_MODEL_NAME}...")
    neighbor_tokenizer = AutoTokenizer.from_pretrained(NEIGHBOR_MODEL_NAME, trust_remote_code=args.trust_remote_code)
    if neighbor_tokenizer.pad_token is None:
        neighbor_tokenizer.pad_token = neighbor_tokenizer.eos_token
    neighbor_tokenizer.padding_side = "left"

    neighbor_model = AutoModelForCausalLM.from_pretrained(
        NEIGHBOR_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=args.trust_remote_code,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    neighbor_model.eval()
    print(f"Neighbor model loaded to: {next(neighbor_model.parameters()).device}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_late_layers_target_modules(start_layer, end_layer, target_keys=None):
    """Generate target_modules for layers [start_layer, end_layer)."""
    if target_keys is None:
        target_keys = ["q_proj", "v_proj"]
    targets = []
    for layer_idx in range(start_layer, end_layer):
        for key in target_keys:
            targets.append(f"model.layers.{layer_idx}.self_attn.{key}")
    return targets


class RollingBinMapper:
    """Maintain rolling quantile bin edges for dynamic binning."""

    def __init__(self, num_bins: int, window_size: int = 500, min_count: int = 100, fixed: bool = False):
        self.num_bins = num_bins
        self.window = deque(maxlen=window_size)
        self.min_count = min_count
        self.fixed = fixed
        self.edges = np.linspace(0.0, 1.0, num_bins + 1)

    def add(self, value: float):
        self.window.append(float(value))
        if not self.fixed and len(self.window) >= self.min_count:
            qs = np.linspace(0.0, 1.0, self.num_bins + 1)
            edges = np.quantile(np.clip(self.window, 0.0, 1.0), qs)
            edges[0] = 0.0
            edges[-1] = 1.0
            self.edges = edges

    def to_bin(self, value: float) -> int:
        v = float(value)
        idx = np.searchsorted(self.edges, v, side="right") - 1
        idx = max(0, min(self.num_bins - 1, idx))
        return int(idx)

    def get_edges(self):
        return self.edges
    
    def is_fixed(self):
        return self.fixed


bin_mapper = RollingBinMapper(NUM_BINS, args.bin_window, args.bin_min_count, fixed=args.fixed_bins)


class PHGate:
    """Page-Hinkley gated TTT trigger with fixed burst and optional cooldown."""
    
    def __init__(self, ema_alpha=0.05, ph_delta=0.05, ph_threshold=4.0, 
                 ttt_burst=20, warmup=50, cooldown=0):
        self.ema_alpha = ema_alpha
        self.ph_delta = ph_delta
        self.ph_threshold = ph_threshold
        self.ttt_burst = ttt_burst
        self.warmup = warmup
        self.cooldown = cooldown
        
        self.ema = None
        self.data = []
        self.segment_start = 0
        self.m_up = self.m_up_min = 0.0
        self.m_down = self.m_down_min = 0.0
        
        self.ttt_remaining = 0
        self.cooldown_remaining = 0
        self.total_triggers = 0
        self.fresh_trigger = False
        self.t = 0
    
    def update(self, entropy: float) -> bool:
        """Update with new entropy value. Returns True if TTT should run.
        Sets self.fresh_trigger = True on the first step of a new trigger."""
        self.t += 1
        self.fresh_trigger = False
        
        if self.ema is None:
            self.ema = entropy
        else:
            self.ema = self.ema_alpha * entropy + (1 - self.ema_alpha) * self.ema
        
        self.data.append(self.ema)
        
        if self.t <= self.warmup:
            return True
        
        if self.cooldown_remaining > 0:
            self.cooldown_remaining -= 1
            return False
        
        segment_data = self.data[self.segment_start:]
        mean_t = np.mean(segment_data)
        
        dev_up = self.ema - mean_t - self.ph_delta
        self.m_up += dev_up
        if len(segment_data) == 1:
            self.m_up_min = self.m_up
        else:
            self.m_up_min = min(self.m_up_min, self.m_up)
        ph_u = self.m_up - self.m_up_min
        
        dev_down = mean_t - self.ema - self.ph_delta
        self.m_down += dev_down
        if len(segment_data) == 1:
            self.m_down_min = self.m_down
        else:
            self.m_down_min = min(self.m_down_min, self.m_down)
        ph_d = self.m_down - self.m_down_min
        
        if ph_u > self.ph_threshold or ph_d > self.ph_threshold:
            self.ttt_remaining = self.ttt_burst
            self.total_triggers += 1
            self.fresh_trigger = True
            self.segment_start = len(self.data)
            self.m_up = self.m_down = 0.0
            self.m_up_min = self.m_down_min = 0.0
        
        if self.ttt_remaining > 0:
            self.ttt_remaining -= 1
            if self.ttt_remaining == 0 and self.cooldown > 0:
                self.cooldown_remaining = self.cooldown
            return True
        return False


def compute_question_entropy(question, model, tokenizer, max_length=256):
    """Compute next-token entropy for a question (single forward pass)."""
    inputs = tokenizer(
        question,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=False
    ).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        last_idx = inputs['attention_mask'].sum(dim=1) - 1
        logits = outputs.logits[0, last_idx, :]
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(log_probs.exp() * log_probs).sum().item()
    
    return entropy


def generate_neighborhood_questions(question, model, tokenizer, n_neighbors=5):
    """Generate neighborhood questions. Simple prompt, minimal parsing."""
    all_questions = []
    
    prompt = f"""Rewrite this question {n_neighbors} different ways:
"{question}"

1."""
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=400,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del outputs
    
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            q = line.lstrip('0123456789').lstrip('.):- ').strip()
            if len(q) > 15:
                all_questions.append({'question': q, 'difficulty': 'similar'})
    
    while len(all_questions) < n_neighbors:
        all_questions.append({'question': question, 'difficulty': 'similar'})
    
    return all_questions[:n_neighbors]


def _is_too_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two answers are too similar (character-level)."""
    a_clean = normalize_answer(a)
    b_clean = normalize_answer(b)
    if not a_clean or not b_clean:
        return True
    if a_clean == b_clean:
        return True
    if a_clean in b_clean or b_clean in a_clean:
        return True
    a_set = set(a_clean)
    b_set = set(b_clean)
    if not a_set or not b_set:
        return True
    overlap = len(a_set & b_set) / max(len(a_set), len(b_set))
    return overlap > threshold



def _is_mc_answer(answer: str) -> str:
    """Check if answer is a single MC letter. Returns the letter or None.
    Handles: 'B', 'B.', 'C. paralysis of the facial muscles...'
    """
    answer_clean = answer.strip().upper().rstrip('.')
    if len(answer_clean) == 1 and answer_clean.isalpha():
        return answer_clean
    m = re.match(r'^([A-Za-z])[\.\)\:]\s', answer.strip())
    if m:
        return m.group(1).upper()
    return None



def generate_distractors(question, base_answer, model, tokenizer, k=2, mc_options=None):
    """Generate k plausible alternative answers for a question.
    
    For MCQ questions: uses the structured mc_options dict directly.
    For open-ended: generates plausible alternatives via the model.
    
    Args:
        mc_options: Structured MCQ options dict {'A': 'text', ...} from dataset.
    """
    base_answer_str = str(base_answer).strip()
    
    mc_letter = _is_mc_answer(base_answer_str)
    if mc_letter and mc_options:
        distractors = []
        for letter in sorted(mc_options.keys()):
            if letter != mc_letter:
                distractors.append(f"{letter}. {mc_options[letter]}")
                if len(distractors) >= k:
                    break
        if distractors:
            return distractors[:k]
    
    is_numeric = bool(re.search(r'^-?\d', base_answer_str))
    
    if is_numeric:
        format_hint = "Use a single short numeric answer (one number) per line. Each answer must be a DIFFERENT number."
    else:
        format_hint = "Keep each answer under 6 words. No explanations."
    
    prompt = f"""Question: {question}

Original answer: {base_answer}

Generate {k} plausible alternative answers to this question that someone might reasonably give.
Requirements:
- Output exactly {k} answers, numbered 1-{k}, one per line.
- Do NOT repeat or paraphrase the original answer.
- Each alternative should be plausible and different from the original.
- {format_hint}"""
    
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            do_sample=True,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    candidates = []
    
    for line in text.splitlines():
        line = line.strip(" -\t")
        if not line:
            continue
        line = re.sub(r"^\d+[\.)]\s*", "", line).strip()
        if not line:
            continue
        
        if _is_too_similar(line, base_answer_str):
            continue
        
        if any(_is_too_similar(line, c) for c in candidates):
            continue
        
        if len(line.split()) > 6 and not is_numeric:
            continue
        
        if is_numeric:
            if not re.match(r"^-?\d", line):
                continue
        
        candidates.append(line)
        if len(candidates) >= k:
            break

    return candidates[:k]


def get_discriminative_confidence(question, answer, reasoning, model, tokenizer, use_p_know=False, claimed_bin=None, candidate_list=None):
    """Get P(True) or P(Know) for the model's judgment."""
    bin_to_pct = {i: (i + 0.5) / NUM_BINS for i in range(NUM_BINS)}
    
    candidates_section = ""
    if candidate_list:
        candidates_lines = []
        for idx, cand in enumerate(candidate_list, 1):
            candidates_lines.append(f"{idx}. {cand}")
        candidates_section = "Here are some brainstormed answers:\n" + "\n".join(candidates_lines) + "\n\n"
    
    if use_p_know:
        if claimed_bin is not None:
            claimed_pct = bin_to_pct.get(claimed_bin, 0.50)
            pct_display = int(claimed_pct * 100)
            verification_prompt = f"""Question: {question}

Are you at least {pct_display}% confident that you know the correct answer to this question?
Please respond with only "True" or "False"."""
        else:
            claimed_pct = 1.0
            verification_prompt = f"""Question: {question}

Are you 100% confident that you know the correct answer to this question?
Please respond with only "True" or "False"."""
    else:
        claimed_pct = None
        verification_prompt = f"""Question: {question}

Proposed Solution:
{reasoning}

Final Answer: {answer}

{candidates_section}\
Is this answer correct? Please respond with only "True" or "False"."""

    messages = [{"role": "user", "content": verification_prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        next_token_logits = outputs.logits[0, -1, :]
    
    true_tokens = tokenizer.encode("True", add_special_tokens=False)
    false_tokens = tokenizer.encode("False", add_special_tokens=False)
    
    true_token_id = true_tokens[0]
    false_token_id = false_tokens[0]
    
    true_logit = next_token_logits[true_token_id].item()
    false_logit = next_token_logits[false_token_id].item()
    
    probs = F.softmax(torch.tensor([true_logit, false_logit]), dim=0)
    p_know = probs[0].item()
    
    if use_p_know and claimed_pct is not None:
        return claimed_pct * p_know
    else:
        return p_know


def p_true_to_bin(p_true):
    """Map P(True) to dynamic confidence bin using rolling quantile edges."""
    return bin_mapper.to_bin(p_true)


def get_bin_digit_tokens(tokenizer):
    """Get token IDs for each confidence bin digit."""
    return {i: tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(NUM_BINS)}


def get_soft_confidence(generated_ids, input_len, model, tokenizer):
    """Extract continuous expected confidence from digit logit distribution."""
    bdt = get_bin_digit_tokens(tokenizer)
    conf_mask = create_confidence_mask(generated_ids[:, input_len:], tokenizer)
    conf_pos = (conf_mask[0] == 1).nonzero(as_tuple=True)[0]
    if len(conf_pos) == 0:
        return None
    with torch.no_grad():
        logits = model(input_ids=generated_ids, attention_mask=torch.ones_like(generated_ids)).logits
    digit_logits = logits[0, input_len + conf_pos[0] - 1][[bdt[b] for b in range(NUM_BINS)]]
    probs = F.softmax(digit_logits, dim=0)
    bins = torch.arange(NUM_BINS, dtype=torch.float32, device=probs.device)
    return ((probs * (bins + 0.5) / NUM_BINS).sum()).item()


def normalize_p_true_sharpened(p_scores, temperature=1.0):
    """Normalize P(True) across candidates with optional sharpening."""
    scores = torch.tensor(p_scores, dtype=torch.float32)
    scores = scores.clamp(min=1e-6)
    temp = max(temperature, 1e-6)
    weights = torch.softmax(torch.log(scores) / temp, dim=0)
    return float(weights[0].item())


def create_confidence_mask(token_ids, tokenizer):
    """Create mask that ONLY targets the confidence bin digit.
    
    Uses token-ID matching instead of character-position alignment to avoid
    fragile char-to-token mapping issues with SentencePiece/BPE tokenizers.
    """
    mask = torch.zeros_like(token_ids, dtype=torch.float)
    
    # Verify the response contains a "Confidence: binN" pattern
    response_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
    conf_match = response_text.lower().find("confidence:")
    if conf_match == -1:
        return mask
    
    conf_section = response_text[conf_match:conf_match+30]
    bin_match = re.search(r'bin(\d+)', conf_section, re.IGNORECASE)
    if not bin_match:
        return mask
    
    target_digit = int(bin_match.group(1))
    bin_digit_tokens = get_bin_digit_tokens(tokenizer)
    if target_digit not in bin_digit_tokens:
        return mask
    target_token_id = bin_digit_tokens[target_digit]
    
    # Search backward from end — "Confidence: binN." is always the last line
    seq_len = len(token_ids[0])
    for i in range(seq_len - 1, max(seq_len - 30, -1), -1):
        if token_ids[0][i].item() == target_token_id:
            mask[0, i] = 1.0
            break
    
    return mask


def compute_calibration_metrics(results, prefix=""):
    """Compute ECE, AUROC, per-bin accuracy."""
    conf_key = f'{prefix}_confidence' if prefix else 'confidence'
    correct_key = f'{prefix}_correct' if prefix else 'correct'
    
    confidences = [r.get(conf_key, 2) for r in results]
    correct = [1 if r.get(correct_key, False) else 0 for r in results]
    conf_probs = [CONF_TO_PROB.get(c, 0.5) for c in confidences]
    
    bin_stats = {}
    for b in range(NUM_BINS):
        bin_mask = [c == b for c in confidences]
        bin_correct = [correct[i] for i, m in enumerate(bin_mask) if m]
        bin_count = sum(bin_mask)
        
        if bin_count > 0:
            bin_acc = sum(bin_correct) / bin_count
            expected_acc = CONF_TO_PROB[b]
            gap = abs(bin_acc - expected_acc)
        else:
            bin_acc = None
            expected_acc = CONF_TO_PROB[b]
            gap = 0
        
        bin_stats[b] = {
            'count': bin_count,
            'accuracy': bin_acc,
            'expected': expected_acc,
            'gap': gap
        }
    
    total_samples = len(results)
    ece = 0.0
    for b, stats in bin_stats.items():
        if stats['count'] > 0 and stats['accuracy'] is not None:
            weight = stats['count'] / total_samples
            ece += weight * stats['gap']
    
    try:
        if sum(correct) > 0 and sum(correct) < len(correct):
            auroc = roc_auc_score(correct, conf_probs)
        else:
            auroc = None
    except Exception:
        auroc = None
    
    conf_distribution = {b: confidences.count(b) for b in range(NUM_BINS)}
    
    return {
        'ece': ece,
        'auroc': auroc,
        'bin_stats': bin_stats,
        'conf_distribution': conf_distribution
    }


def get_baseline_answer(question, model, tokenizer, return_ids=False):
    """Get answer without test-time training."""
    prompt = create_qa_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    
    input_len = inputs['input_ids'].shape[1]
    response = tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
    
    result = {
        'answer': extract_answer(response),
        'confidence': extract_confidence(response),
        'response': response
    }
    if return_ids:
        result['generated_ids'] = outputs
        result['input_len'] = input_len
    else:
        del outputs
    
    return result


# =============================================================================
# TEMPERATURE SCALING
# =============================================================================

def apply_temperature_scaling(results, temperature=1.0):
    """
    Apply temperature scaling to confidence bins.
    
    Temperature scaling maps confidence probabilities through a temperature parameter:
    - T < 1: Sharpens (more confident)
    - T = 1: No change
    - T > 1: Smooths (less confident)
    """
    scaled_results = []
    for r in results:
        original_conf = r['confidence']
        original_prob = CONF_TO_PROB[original_conf]
        
        if temperature != 1.0:
            # Map to logit space, scale, map back
            epsilon = 1e-6
            p = max(epsilon, min(1 - epsilon, original_prob))
            logit = np.log(p / (1 - p))
            scaled_logit = logit / temperature
            scaled_prob = 1 / (1 + np.exp(-scaled_logit))
            
            # Map back to bin
            scaled_bin = bin_mapper.to_bin(scaled_prob)
        else:
            scaled_bin = original_conf
        
        r_scaled = r.copy()
        r_scaled['confidence'] = scaled_bin
        scaled_results.append(r_scaled)
    
    return scaled_results


def find_optimal_temperature(val_results, temperature_range=None):
    """
    Find optimal temperature on validation set by minimizing ECE.
    
    Args:
        val_results: List of results with 'confidence' and 'correct' keys
        temperature_range: List of temperatures to try
    
    Returns:
        Best temperature value
    """
    if temperature_range is None:
        temperature_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]
    
    best_temp = 1.0
    best_ece = float('inf')
    temp_results = []
    
    for temp in temperature_range:
        scaled_results = apply_temperature_scaling(val_results, temperature=temp)
        metrics = compute_calibration_metrics(scaled_results, prefix="")
        ece = metrics['ece']
        temp_results.append((temp, ece))
        
        if ece < best_ece:
            best_ece = ece
            best_temp = temp
    
    # Print all results with best marked
    for temp, ece in temp_results:
        marker = " *BEST*" if temp == best_temp else ""
        print(f"  Temp={temp:.2f} → ECE={ece:.4f}{marker}")
    
    print(f"\nOptimal temperature: T={best_temp:.2f} (ECE={best_ece:.4f})")
    return best_temp


# =============================================================================
# DISCRIMINATIVE TTT TRAINING
# =============================================================================

def compute_sc_agreement(question, greedy_answer, model, tokenizer, n_samples, temperature, dataset_type="auto"):
    """Compute unsupervised Self-Consistency: fraction of temperature samples agreeing with greedy answer."""
    prompt = create_qa_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    agree = 0
    for _ in range(n_samples):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tokenizer.eos_token_id,
            )
        resp = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        sampled_ans = extract_answer(resp)
        if answers_match(sampled_ans, {'ground_truth_answer': greedy_answer}, dataset_type=dataset_type):
            agree += 1
        del out
    return agree / n_samples


def train_single_question_discriminative(question, train_model, optimizer, tokenizer, n_neighbors=5, n_epochs=3, dataset_type="auto", neighbors_only=False, mc_options=None, nb_model=None, nb_tokenizer=None, ranking_buffer=None):
    """Train on a single question using discriminative P(True) as pseudo-label.
    
    Args:
        mc_options: Structured MCQ options dict {'A': 'text', ...} from dataset.
                    Used directly for P(True) normalization instead of regex parsing.
    """
    
    bin_digit_tokens = get_bin_digit_tokens(tokenizer)
    pknow_values = []
    
    questions_to_train = []
    if not neighbors_only:
        questions_to_train.append({'question': question, 'difficulty': 'input'})
    
    train_model.eval()
    if n_neighbors > 0:
        _nb_model = nb_model if nb_model is not None else neighbor_model
        _nb_tok = nb_tokenizer if nb_tokenizer is not None else neighbor_tokenizer
        neighbors = generate_neighborhood_questions(question, _nb_model, _nb_tok, n_neighbors)
        questions_to_train.extend(neighbors)
    
    if len(questions_to_train) == 0:
        questions_to_train.append({'question': question, 'difficulty': 'input'})
    
    # === Pass 1: Collect raw signals for all training questions ===
    # When use_base_target: compute generation + P(True) with base model (no adapter)
    # so the target bin matches the baseline's calibration signal
    if args.use_base_target:
        train_model.disable_adapter_layers()
    try:
        neighbor_data = []
        for n in questions_to_train:
            q = n['question'] if isinstance(n, dict) else n
            difficulty = n.get('difficulty', 'similar') if isinstance(n, dict) else 'similar'
            
            # When reusing MCQ options, append choices to neighbor questions
            # so the model answers with a letter (consistent with original MCQ format)
            if mc_options and args.reuse_mcq_options and difficulty != 'input':
                choices = "\n".join(f"{ltr}. {mc_options[ltr]}" for ltr in sorted(mc_options))
                q = f"{q}\n{choices}"
            
            prompt = create_qa_prompt(q)
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
            
            with torch.no_grad():
                outputs = train_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            answer = extract_answer(response)
            claimed_conf = extract_confidence(response)
            
            p_know = get_discriminative_confidence(
                q,
                answer,
                response,
                train_model,
                tokenizer,
                use_p_know=True,
                claimed_bin=claimed_conf
            )
            pknow_values.append(p_know)
            del outputs
            
            if args.use_sc_target:
                sc_score = compute_sc_agreement(
                    q, answer, train_model, tokenizer,
                    n_samples=args.sc_n_samples,
                    temperature=args.sc_temperature,
                    dataset_type=dataset_type,
                )
                p_true_raw = sc_score
                p_true_norm = sc_score
                no_distractors = False
            else:
                # Use structured mc_options when the answer is a valid MCQ letter;
                # otherwise fall back to distractor generation.
                use_mc = mc_options if (mc_options and (args.reuse_mcq_options or difficulty == 'input')) else None
                ans_letter = _is_mc_answer(answer) if use_mc else None
                if use_mc and ans_letter and ans_letter in use_mc:
                    candidates = [f"{ans_letter}. {use_mc[ans_letter]}"]
                    for ltr in sorted(use_mc.keys()):
                        if ltr != ans_letter:
                            candidates.append(f"{ltr}. {use_mc[ltr]}")
                else:
                    distractors = generate_distractors(q, answer, train_model, tokenizer, k=args.k_distractors, mc_options=mc_options)
                    candidates = [answer] + distractors
                
                no_distractors = len(candidates) <= 1
                
                p_scores = []
                with torch.no_grad():
                    for cand in candidates:
                        p_cand = get_discriminative_confidence(
                            q,
                            cand,
                            response,
                            train_model,
                            tokenizer,
                            use_p_know=False,
                            claimed_bin=None,
                            candidate_list=candidates,
                        )
                        p_scores.append(p_cand)
                p_true_raw = p_scores[0]
                
                if no_distractors:
                    p_true_norm = p_true_raw
                else:
                    p_true_norm = normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
            
            neighbor_data.append({
                'question': q,
                'difficulty': difficulty,
                'answer': answer,
                'response': response,
                'p_true_raw': p_true_raw,
                'p_true_norm': p_true_norm,
                'p_know': p_know,
                'no_distractors': no_distractors,
            })
    finally:
        # Re-enable adapter after Pass 1 so training uses the adapted model
        if args.use_base_target:
            train_model.enable_adapter_layers()
    
    # === Pass 2: Compute derived values with global p_know_mean ===
    p_know_mean = float(np.mean(pknow_values)) if pknow_values else 0.5
    
    for nd in neighbor_data:
        nd['p_know_mean'] = p_know_mean
        nd['p_fused'] = nd['p_true_norm'] * p_know_mean
        
        if args.use_raw_ptrue or nd['no_distractors']:
            p_for_bin = nd['p_true_raw']
        elif args.use_p_know:
            p_for_bin = nd['p_know']
        elif args.use_fused_confidence:
            p_for_bin = nd['p_fused']
        else:
            p_for_bin = nd['p_true_norm']

        if nd.get('difficulty') == 'input':
            bin_mapper.add(p_for_bin)
        nd['target_bin'] = p_true_to_bin(p_for_bin)
    
    train_model.train()
    training_losses = []
    
    generation_kwargs = {
        "max_new_tokens": 200,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
    }
    
    for epoch in range(n_epochs):
        epoch_losses = []
        optimizer.zero_grad()
        valid_samples = 0
        
        for nd in neighbor_data:
            prompt = create_qa_prompt(nd['question'])
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
            
            max_retries = 3
            outputs = None
            conf_mask = None
            
            for retry in range(max_retries):
                with torch.no_grad():
                    if retry == 0:
                        outputs = train_model.generate(**inputs, **generation_kwargs)
                    else:
                        outputs = train_model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                
                response_tokens = outputs[:, inputs['input_ids'].shape[1]:].clone()
                conf_mask = create_confidence_mask(response_tokens, tokenizer)
                
                if conf_mask.sum() > 0:
                    break
                
                del outputs, response_tokens, conf_mask
                torch.cuda.empty_cache()
                outputs = None
            
            if outputs is None:
                continue
            
            if args.use_ranking_loss:
                # Pairwise ranking loss against buffer of past questions
                conf_pos = (conf_mask[0] == 1).nonzero(as_tuple=True)[0]
                if len(conf_pos) == 0:
                    del outputs
                    continue
                model_output = train_model(input_ids=outputs, attention_mask=torch.ones_like(outputs))
                logit_pos = inputs['input_ids'].shape[1] + conf_pos[0] - 1
                digit_logits = model_output.logits[0, logit_pos][[bin_digit_tokens[b] for b in range(NUM_BINS)]]
                digit_probs = F.softmax(digit_logits, dim=0)
                bin_vals = torch.arange(NUM_BINS, dtype=torch.float32, device=digit_probs.device)
                expected_conf = (digit_probs * (bin_vals + 0.5) / NUM_BINS).sum()
                
                if ranking_buffer:
                    cur_ptrue = nd['p_true_norm']
                    loss = torch.tensor(0.0, device=expected_conf.device)
                    n_pairs = 0
                    for buf_conf, buf_ptrue in ranking_buffer:
                        if abs(cur_ptrue - buf_ptrue) < 0.05:
                            continue
                        diff = expected_conf - buf_conf
                        if cur_ptrue > buf_ptrue:
                            loss = loss - F.logsigmoid(diff)
                        else:
                            loss = loss - F.logsigmoid(-diff)
                        n_pairs += 1
                    loss = loss / n_pairs if n_pairs > 0 else None
                else:
                    loss = None
            elif args.use_mse_loss:
                # MSE loss: expected confidence from logits vs P(True) norm
                conf_pos = (conf_mask[0] == 1).nonzero(as_tuple=True)[0]
                if len(conf_pos) == 0:
                    del outputs
                    continue
                full_input = outputs
                model_output = train_model(input_ids=full_input, attention_mask=torch.ones_like(full_input))
                logit_pos = inputs['input_ids'].shape[1] + conf_pos[0] - 1
                logits_at_conf = model_output.logits[0, logit_pos]
                digit_token_ids = [bin_digit_tokens[i] for i in range(NUM_BINS)]
                digit_logits = logits_at_conf[digit_token_ids]
                digit_probs = F.softmax(digit_logits, dim=0)
                bin_values = torch.arange(NUM_BINS, dtype=torch.float32, device=digit_probs.device)
                expected_conf = (digit_probs * (bin_values + 0.5) / NUM_BINS).sum()
                target_ptrue = torch.tensor(nd['p_true_norm'], dtype=torch.float32, device=expected_conf.device)
                if args.use_directional_target:
                    alpha = max(0.0, min(1.0, float(args.directional_alpha)))
                    max_step = max(0.0, float(args.directional_clip))
                    delta = torch.clamp(target_ptrue - expected_conf.detach(), min=-max_step, max=max_step)
                    target_conf = expected_conf.detach() + alpha * delta
                    loss = (expected_conf - target_conf) ** 2
                else:
                    loss = (expected_conf - target_ptrue) ** 2
            else:
                labels = outputs.clone()
                labels[:, :inputs['input_ids'].shape[1]] = -100
                
                response_labels = labels[:, inputs['input_ids'].shape[1]:]
                target_token_id = bin_digit_tokens[nd['target_bin']]
                
                for i in range(response_tokens.shape[1]):
                    if conf_mask[0, i] == 1:
                        response_labels[0, i] = target_token_id
                    else:
                        response_labels[0, i] = -100
                
                labels[:, inputs['input_ids'].shape[1]:] = response_labels
                
                model_output = train_model(
                    input_ids=outputs,
                    attention_mask=torch.ones_like(outputs),
                    labels=labels
                )
                
                loss = model_output.loss
            if loss is not None and not torch.isnan(loss):
                if args.kl_reg_beta > 0:
                    input_len = inputs['input_ids'].shape[1]
                    adapted_logits = model_output.logits[:, input_len:, :]
                    with torch.no_grad(), train_model.disable_adapter():
                        base_output = train_model(input_ids=outputs,
                                                   attention_mask=torch.ones_like(outputs))
                    base_logits = base_output.logits[:, input_len:, :].detach()
                    kl = F.kl_div(
                        F.log_softmax(adapted_logits, dim=-1),
                        F.softmax(base_logits, dim=-1),
                        reduction="batchmean"
                    )
                    loss = loss + args.kl_reg_beta * kl
                    del base_output, base_logits, adapted_logits

                loss.backward()
                valid_samples += 1
                epoch_losses.append(loss.item())
            
            del outputs
        
        if valid_samples > 0:
            for param in train_model.parameters():
                if param.grad is not None:
                    param.grad.div_(valid_samples)
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
            optimizer.step()
        else:
            optimizer.zero_grad()
        
        training_losses.append(np.mean(epoch_losses) if epoch_losses else 0.0)
    
    # Update ranking buffer with the input question's data
    if ranking_buffer is not None:
        input_nd = next((nd for nd in neighbor_data if nd.get('difficulty') == 'input'), None)
        if input_nd:
            ranking_buffer.append((CONF_TO_PROB.get(input_nd['target_bin'], 0.5), input_nd['p_true_norm']))
            if len(ranking_buffer) > args.ranking_buffer_size:
                ranking_buffer.pop(0)
    
    train_model.eval()
    prompt = create_qa_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
    
    with torch.no_grad():
        outputs = train_model.generate(**inputs, **generation_kwargs)
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    
    answer = extract_answer(response)
    claimed_conf = extract_confidence(response)
    
    # Soft confidence: read expected value from digit logit distribution
    if args.use_soft_confidence:
        soft = get_soft_confidence(outputs, inputs['input_ids'].shape[1], train_model, tokenizer)
    del outputs
    
    with torch.no_grad():
        final_p_true = get_discriminative_confidence(question, answer, response, train_model, tokenizer, 
                                                     use_p_know=args.use_p_know, claimed_bin=claimed_conf)
    
    # When report_ptrue: compute P(True) norm and use its bin as confidence
    final_conf = p_true_to_bin(soft) if (args.use_soft_confidence and soft is not None) else claimed_conf
    final_p_true_norm = final_p_true
    if args.report_ptrue:
        use_mc = mc_options if mc_options else None
        ans_letter = _is_mc_answer(answer) if use_mc else None
        if use_mc and ans_letter and ans_letter in use_mc:
            cands = [f"{ans_letter}. {use_mc[ans_letter]}"] + [f"{l}. {use_mc[l]}" for l in sorted(use_mc.keys()) if l != ans_letter]
        else:
            cands = [answer] + generate_distractors(question, answer, train_model, tokenizer, k=args.k_distractors, mc_options=mc_options)
        if len(cands) > 1:
            with torch.no_grad():
                ps = [get_discriminative_confidence(question, c, response, train_model, tokenizer, use_p_know=False, claimed_bin=None, candidate_list=cands) for c in cands]
            final_p_true_norm = normalize_p_true_sharpened(ps, temperature=args.norm_temperature)
        final_conf = p_true_to_bin(final_p_true_norm)
    
    neighbor_p_know_mean = float(np.mean([n.get('p_know', 0.5) for n in neighbor_data])) if neighbor_data else 0.5
    neighbor_p_fused_mean = float(np.mean([n.get('p_fused', n.get('p_true_norm', 0.5)) for n in neighbor_data])) if neighbor_data else 0.5
    
    return {
        'answer': answer,
        'confidence': final_conf,
        'response': response,
        'neighbors': neighbor_data,
        'training_losses': training_losses,
        'p_true': final_p_true,
        'p_true_norm': final_p_true_norm,
        'neighbor_p_know_mean': neighbor_p_know_mean,
        'neighbor_p_fused_mean': neighbor_p_fused_mean
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_continual_experiment():
    """Run continual TTT experiment across multiple domains."""
    
    if args.mode == 'custom' and args.domains:
        domain_order = args.domains
    else:
        domain_order = BENCHMARK_MODES[args.mode]
    
    no_accumulation = not args.accumulation
    use_wandb = not args.no_wandb
    os.makedirs("results", exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RUNNING CONTINUAL DISCRIMINATIVE TTT - LAYERS {LAYER_START}-{LAYER_END-1}")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Domain order: {' → '.join(domain_order)}")
    print(f"Questions per domain: {args.questions_per_domain}")
    print(f"Neighbors per question: {args.n_neighbors}")
    print(f"Epochs per question: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight accumulation: {'OFF' if no_accumulation else 'ON'}")
    print(f"Baseline only: {args.baseline_only}")
    bin_mode = "FIXED (uniform edges)" if args.fixed_bins else f"ADAPTIVE (window={args.bin_window}, min={args.bin_min_count})"
    print(f"Bin edges: {bin_mode}")
    if args.use_ph_gate:
        print(f"PH Gate: ON (delta={args.ph_delta}, threshold={args.ph_threshold}, burst={args.ttt_burst})")
    print()
    
    print("Loading datasets...")
    all_data = {}
    for domain in set(domain_order):
        config = DOMAIN_CONFIG[domain]
        data = load_qa_dataset(
            dataset_mode=config['dataset_mode'],
            num_samples=args.questions_per_domain * domain_order.count(domain)
        )
        for q in data:
            q['domain'] = domain
            q['dataset_type'] = config['dataset_type']
        all_data[domain] = data
        print(f"  {config['name']}: {len(data)} questions")
    
    if args.mode == 'interleaved':
        questions = []
        domain_indices = {d: 0 for d in set(domain_order)}
        max_per_domain = args.questions_per_domain
        
        while any(domain_indices[d] < min(len(all_data[d]), max_per_domain) for d in set(domain_order)):
            for domain in set(domain_order):
                if domain_indices[domain] < min(len(all_data[domain]), max_per_domain):
                    questions.append(all_data[domain][domain_indices[domain]])
                    domain_indices[domain] += 1
        
        random.shuffle(questions)
    else:
        questions = []
        domain_usage = defaultdict(int)
        for domain in domain_order:
            start_idx = domain_usage[domain] * args.questions_per_domain
            end_idx = start_idx + args.questions_per_domain
            questions.extend(all_data[domain][start_idx:end_idx])
            domain_usage[domain] += 1
    
    print(f"Total questions: {len(questions)}")
    
    # =============================================================================
    # TEMPERATURE SCALING MODE
    # =============================================================================
    
    if args.temperature_scaling:
        print(f"\n{'='*70}")
        print("RUNNING TEMPERATURE SCALING BASELINE")
        print(f"{'='*70}\n")
        
        # Split: first 100 questions per domain for validation, rest for test
        val_questions = []
        test_questions = []
        
        for domain in set(domain_order):
            domain_qs = [q for q in questions if q['domain'] == domain]
            val_size = min(100, len(domain_qs) // 4)  # Use 25% or 100, whichever is smaller
            val_questions.extend(domain_qs[:val_size])
            test_questions.extend(domain_qs[val_size:])
        
        print(f"Validation set: {len(val_questions)} questions")
        print(f"Test set: {len(test_questions)} questions\n")
        
        # Get baseline predictions on validation set
        print("Phase 1: Calibrating temperature on validation set...")
        val_results = []
        for qa in tqdm(val_questions, desc="Validation"):
            baseline = get_baseline_answer(qa['question'], base_model, tokenizer)
            is_correct = answers_match(baseline['answer'], qa, dataset_type=qa['dataset_type'])
            val_results.append({
                'confidence': baseline['confidence'],
                'correct': is_correct
            })
        
        # Find optimal temperature
        print("\nSearching for optimal temperature...")
        optimal_temp = find_optimal_temperature(val_results)
        
        # Apply to test set
        print(f"\nPhase 2: Applying T={optimal_temp:.2f} to test set...")
        test_results = []
        domain_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'brier_sum': 0})
        
        for qa in tqdm(test_questions, desc="Testing"):
            baseline = get_baseline_answer(qa['question'], base_model, tokenizer)
            is_correct = answers_match(baseline['answer'], qa, dataset_type=qa['dataset_type'])
            
            domain = qa['domain']
            domain_stats[domain]['correct'] += int(is_correct)
            domain_stats[domain]['total'] += 1
            
            test_results.append({
                'question': qa['question'][:200],
                'ground_truth': qa['ground_truth_answer'],
                'domain': domain,
                'answer': baseline['answer'],
                'confidence': baseline['confidence'],
                'correct': is_correct
            })
        
        # Apply temperature scaling
        scaled_results = apply_temperature_scaling(test_results, temperature=optimal_temp)
        
        # Recompute brier scores after scaling
        for r in scaled_results:
            conf_prob = CONF_TO_PROB[r['confidence']]
            r['brier'] = (conf_prob - float(r['correct'])) ** 2
            domain_stats[r['domain']]['brier_sum'] += r['brier']
        
        # Compute metrics
        final_metrics = compute_calibration_metrics(scaled_results, prefix="")
        
        # Per-domain metrics
        domain_metrics = {}
        for domain in set([r['domain'] for r in scaled_results]):
            domain_results = [r for r in scaled_results if r['domain'] == domain]
            domain_cal = compute_calibration_metrics(domain_results, prefix="")
            domain_metrics[domain] = {
                'acc': domain_stats[domain]['correct'] / domain_stats[domain]['total'],
                'brier': domain_stats[domain]['brier_sum'] / domain_stats[domain]['total'],
                'ece': domain_cal['ece'],
                'auroc': domain_cal['auroc'],
                'count': domain_stats[domain]['total']
            }
        
        # Print results
        print(f"\n{'='*70}")
        print("TEMPERATURE SCALING RESULTS")
        print(f"{'='*70}")
        print(f"Optimal Temperature: {optimal_temp:.2f}")
        print(f"Overall ECE: {final_metrics['ece']:.4f}")
        if final_metrics['auroc']:
            print(f"Overall AUROC: {final_metrics['auroc']:.4f}")
        
        total_correct = sum(d['correct'] for d in domain_stats.values())
        total_count = sum(d['total'] for d in domain_stats.values())
        overall_acc = total_correct / total_count
        overall_brier = sum(d['brier_sum'] for d in domain_stats.values()) / total_count
        
        print(f"Overall Accuracy: {total_correct}/{total_count} = {overall_acc:.1%}")
        print(f"Overall Brier: {overall_brier:.4f}")
        
        print("\nPer-Domain Results:")
        for domain in domain_metrics:
            m = domain_metrics[domain]
            auroc_str = f"{m['auroc']:.4f}" if m['auroc'] else "N/A"
            print(f"  {DOMAIN_CONFIG[domain]['name']:12} Acc={m['acc']:.1%}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}  AUROC={auroc_str}")
        
        print(f"\nPer-bin accuracy:")
        for b, stats in final_metrics['bin_stats'].items():
            if stats['count'] > 0:
                acc_str = f"{stats['accuracy']:.1%}" if stats['accuracy'] is not None else "N/A"
                print(f"  Bin{b} (expect {stats['expected']:.0%}): {acc_str} ({stats['count']} samples)")
        
        # Save results
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.run_name or "temp_scaling",
                config={
                    'method': 'temperature_scaling',
                    'optimal_temperature': optimal_temp,
                    'val_size': len(val_questions),
                    'test_size': len(test_questions)
                }
            )
            
            wandb.log({
                'final/ece': final_metrics['ece'],
                'final/auroc': final_metrics['auroc'],
                'final/acc': overall_acc,
                'final/brier': overall_brier,
                'optimal_temperature': optimal_temp
            })
            
            for domain in domain_metrics:
                m = domain_metrics[domain]
                wandb.log({
                    f'final/{domain}_ece': m['ece'],
                    f'final/{domain}_auroc': m['auroc'],
                    f'final/{domain}_acc': m['acc'],
                    f'final/{domain}_brier': m['brier']
                })
            
            wandb.finish()
        
        # Save to file
        name_suffix = args.run_name if args.run_name else 'temp_scaling'
        output_file = f"results/temp_scaling_{name_suffix}_results.json"
        with open(output_file, 'w') as f:
            json.dump({
                'config': {
                    'method': 'temperature_scaling',
                    'optimal_temperature': optimal_temp,
                    'val_size': len(val_questions),
                    'test_size': len(test_questions)
                },
                'final_metrics': {
                    'ece': final_metrics['ece'],
                    'auroc': final_metrics['auroc'],
                    'accuracy': overall_acc,
                    'brier': overall_brier
                },
                'domain_metrics': domain_metrics,
                'results': scaled_results
            }, f, indent=2, default=str)
        
        print(f"\nResults saved to {output_file}")
        print(f"{'='*70}\n")
        
        return scaled_results
    
    # =============================================================================
    # NORMAL TTT MODE (Continue with existing code)
    # =============================================================================
    
    if use_wandb:
        if args.run_name:
            run_name = args.run_name
        else:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            accum_str = "noaccum" if no_accumulation else "accum"
            baseline_str = "BASELINE_" if args.baseline_only else ""
            run_name = f"{baseline_str}CONT_{args.mode}_{accum_str}_nb{args.n_neighbors}_{timestamp}"
        
        wandb_config = vars(args).copy()
        wandb_config["method"] = "continual_discriminative_late_layers"
        wandb_config["domains"] = domain_order
        wandb_config["target_layers"] = f"{LAYER_START}-{LAYER_END-1}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            settings=wandb.Settings(console="off"),
            config=wandb_config,
        )
        print(f"W&B run: {run_name}")
    
    lora_keys = [k.strip() for k in args.lora_target_keys.split(",")]
    target_modules = get_late_layers_target_modules(LAYER_START, LAYER_END, target_keys=lora_keys)
    print(f"Target modules ({len(target_modules)}): {target_modules[:4]}...")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    train_model = get_peft_model(base_model, lora_config)
    train_model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr) if not args.baseline_only else None
    
    results = []
    domain_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'brier_sum': 0})
    global_correct = 0
    global_brier_sum = 0
    
    learning_curve = {
        'step': [],
        'global_acc': [],
        'global_brier': [],
        'global_ece': [],
        'avg_loss': [],
        'avg_p_true': []
    }
    
    ranking_buffer = [] if args.use_ranking_loss else None
    
    ph_gate = None
    if args.use_ph_gate:
        ph_gate = PHGate(
            ema_alpha=args.ema_alpha,
            ph_delta=args.ph_delta,
            ph_threshold=args.ph_threshold,
            ttt_burst=args.ttt_burst,
            warmup=args.ttt_warmup,
            cooldown=args.ph_cooldown
        )
        print(f"PH Gate enabled: burst={args.ttt_burst}, warmup={args.ttt_warmup}, "
              f"delta={args.ph_delta}, threshold={args.ph_threshold}, ema_alpha={args.ema_alpha}, "
              f"cooldown={args.ph_cooldown}, reset_on_trigger={args.ph_reset_on_trigger}")
    
    for i, qa in enumerate(tqdm(questions, desc=f"Continual TTT ({args.mode})")):
        question = qa['question']
        ground_truth = qa['ground_truth_answer']
        domain = qa['domain']
        dataset_type = qa['dataset_type']
        
        should_do_ttt = True
        if args.use_ph_gate and not args.baseline_only:
            with train_model.disable_adapter():
                entropy = compute_question_entropy(question, train_model, tokenizer)
            should_do_ttt = ph_gate.update(entropy)
            if args.ph_reset_on_trigger and ph_gate.fresh_trigger:
                with torch.no_grad():
                    for name, param in train_model.named_parameters():
                        if 'lora_A' in name:
                            torch.nn.init.kaiming_uniform_(param, a=5**0.5)
                        elif 'lora_B' in name:
                            param.zero_()
                optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr)
                print(f"  [PH RESET] Trigger #{ph_gate.total_triggers} at step {i+1} — LoRA zeroed, optimizer reset")
        
        if no_accumulation and not args.baseline_only and should_do_ttt:
            with torch.no_grad():
                for name, param in train_model.named_parameters():
                    if 'lora_A' in name:
                        torch.nn.init.kaiming_uniform_(param, a=5**0.5)
                    elif 'lora_B' in name:
                        param.zero_()
            optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr)
        
        with train_model.disable_adapter():
            baseline = get_baseline_answer(question, train_model, tokenizer, return_ids=args.use_soft_confidence)
        baseline_correct = answers_match(baseline['answer'], qa, dataset_type=dataset_type)
        
        if args.use_bin_gate and not args.baseline_only and should_do_ttt:
            # Generate with adapted model to get verbalized bin
            _gp = create_qa_prompt(question)
            _gi = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": _gp}], tokenize=False, add_generation_prompt=True), return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
            with torch.no_grad():
                _go = train_model.generate(**_gi, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
            _gr = tokenizer.decode(_go[0][_gi['input_ids'].shape[1]:], skip_special_tokens=True)
            _ga, _gv = extract_answer(_gr), extract_confidence(_gr)
            del _go
            # Compute P(True) — base model if use_base_target, else adapted
            mc_opts = qa.get('mc_options')
            ans_letter = _is_mc_answer(_ga) if mc_opts else None
            if mc_opts and ans_letter and ans_letter in mc_opts:
                candidates = [f"{ans_letter}. {mc_opts[ans_letter]}"] + [f"{l}. {mc_opts[l]}" for l in sorted(mc_opts.keys()) if l != ans_letter]
            else:
                candidates = [_ga] + generate_distractors(question, _ga, train_model, tokenizer, k=args.k_distractors, mc_options=mc_opts)
            _ctx = train_model.disable_adapter() if args.use_base_target else nullcontext()
            with _ctx, torch.no_grad():
                p_scores = [get_discriminative_confidence(question, c, _gr, train_model, tokenizer, use_p_know=False, claimed_bin=None, candidate_list=candidates) for c in candidates]
            _pn = p_scores[0] if len(candidates) <= 1 else normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
            _gap = _gv - p_true_to_bin(_pn)
            if args.overconfident_only:
                if _gap <= args.bin_gate_threshold:
                    should_do_ttt = False
            else:
                if abs(_gap) <= args.bin_gate_threshold:
                    should_do_ttt = False
        
        if args.baseline_only:
            is_correct = baseline_correct

            with torch.no_grad():
                baseline_p_true = get_discriminative_confidence(
                    question, baseline['answer'], baseline['response'], train_model, tokenizer,
                    use_p_know=args.use_p_know, claimed_bin=baseline['confidence']
                )

            if args.use_soft_confidence and baseline.get('generated_ids') is not None:
                with train_model.disable_adapter(), torch.no_grad():
                    soft_conf = get_soft_confidence(baseline['generated_ids'], baseline['input_len'], train_model, tokenizer)
                del baseline['generated_ids']
                if soft_conf is not None:
                    baseline_conf_bin = p_true_to_bin(soft_conf)
                else:
                    baseline_conf_bin = baseline['confidence']
                conf_prob = CONF_TO_PROB.get(baseline_conf_bin, 0.5)
                brier = (conf_prob - float(is_correct)) ** 2
                brier_p_true = (baseline_p_true - float(is_correct)) ** 2
                baseline_conf_for_log = baseline_conf_bin
                neighbor_p_trues = []
                avg_p_true = baseline_p_true
            elif args.baseline_use_ptrue_norm:
                mc_opts = qa.get('mc_options')
                ans_letter = _is_mc_answer(baseline['answer']) if mc_opts else None
                if mc_opts and ans_letter and ans_letter in mc_opts:
                    candidates = [f"{ans_letter}. {mc_opts[ans_letter]}"]
                    for ltr in sorted(mc_opts.keys()):
                        if ltr != ans_letter:
                            candidates.append(f"{ltr}. {mc_opts[ltr]}")
                else:
                    distractors = generate_distractors(question, baseline['answer'], train_model, tokenizer, k=args.k_distractors, mc_options=mc_opts)
                    candidates = [baseline['answer']] + distractors
                
                no_distractors = len(candidates) <= 1
                
                with torch.no_grad():
                    p_scores = []
                    for cand in candidates:
                        p_cand = get_discriminative_confidence(
                            question,
                            cand,
                            baseline['response'],
                            train_model,
                            tokenizer,
                            use_p_know=False,
                            claimed_bin=None,
                            candidate_list=candidates,
                        )
                        p_scores.append(p_cand)
                p_true_raw = p_scores[0]
                
                if no_distractors:
                    p_true_norm = p_true_raw
                else:
                    p_true_norm = normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
                
                bin_mapper.add(p_true_norm)
                baseline_conf_bin = p_true_to_bin(p_true_norm)
                conf_prob = CONF_TO_PROB.get(baseline_conf_bin, 0.5)
                brier = (conf_prob - float(is_correct)) ** 2
                brier_p_true = (p_true_norm - float(is_correct)) ** 2
                baseline_conf_for_log = baseline_conf_bin
                neighbor_p_trues = [p_true_norm]
                avg_p_true = p_true_norm
            else:
                conf_prob = CONF_TO_PROB.get(baseline['confidence'], 0.5)
                brier = (conf_prob - float(is_correct)) ** 2
                brier_p_true = (baseline_p_true - float(is_correct)) ** 2
                baseline_conf_for_log = baseline['confidence']
                neighbor_p_trues = []
                avg_p_true = baseline_p_true
            
            result = {
                'question': question[:200],
                'ground_truth': ground_truth,
                'domain': domain,
                'baseline_answer': baseline['answer'],
                'baseline_confidence': baseline_conf_for_log,
                'baseline_p_true': baseline_p_true,
                'baseline_correct': is_correct,
                'ttt_answer': baseline['answer'],
                'ttt_confidence': baseline_conf_for_log,
                'ttt_correct': is_correct,
                'ttt_p_true': baseline_p_true,
                'training_losses': [],
                'neighbor_p_trues': neighbor_p_trues,
                'avg_p_true': avg_p_true,
                'brier_p_true': brier_p_true
            }
            avg_loss = 0.0
            
        elif should_do_ttt:
            ttt_result = train_single_question_discriminative(
                question, train_model, optimizer, tokenizer,
                n_neighbors=args.n_neighbors, n_epochs=args.n_epochs, dataset_type=dataset_type,
                neighbors_only=args.neighbors_only, mc_options=qa.get('mc_options'),
                nb_model=neighbor_model, nb_tokenizer=neighbor_tokenizer,
                ranking_buffer=ranking_buffer
            )
            
            is_correct = answers_match(ttt_result['answer'], qa, dataset_type=dataset_type)
            
            conf_prob = CONF_TO_PROB.get(ttt_result['confidence'], 0.5)
            brier = (conf_prob - float(is_correct)) ** 2
            
            if args.use_fused_confidence:
                neighbor_p_trues = [n.get('p_fused', n.get('p_true_norm', 0.5)) for n in ttt_result['neighbors']]
            else:
                neighbor_p_trues = [n.get('p_true_norm', 0.5) for n in ttt_result['neighbors']]
            avg_p_true = np.mean(neighbor_p_trues) if neighbor_p_trues else 0.5
            avg_loss = np.mean(ttt_result['training_losses']) if ttt_result['training_losses'] else 0.0
            
            brier_p_true = (ttt_result['p_true'] - float(is_correct)) ** 2
            
            result = {
                'question': question[:200],
                'ground_truth': ground_truth,
                'domain': domain,
                'baseline_answer': baseline['answer'],
                'baseline_confidence': baseline['confidence'],
                'baseline_correct': baseline_correct,
                'ttt_answer': ttt_result['answer'],
                'ttt_confidence': ttt_result['confidence'],
                'ttt_correct': is_correct,
                'ttt_p_true': ttt_result['p_true'],
                'ttt_p_know_mean': ttt_result.get('neighbor_p_know_mean', 0.5),
                'ttt_p_fused_mean': ttt_result.get('neighbor_p_fused_mean', avg_p_true),
                'brier_p_true': brier_p_true,
                'training_losses': ttt_result['training_losses'],
                'neighbor_p_trues': neighbor_p_trues,
                'avg_p_true': avg_p_true,
                'ttt_skipped': False
            }
        
        else:
            # Skipped (PH or bin gate): no training, but use adapted model for readout
            if args.use_soft_confidence:
                # Generate with adapted model (LoRA enabled) and read soft confidence
                _sp = create_qa_prompt(question)
                _si = tokenizer(tokenizer.apply_chat_template([{"role": "user", "content": _sp}], tokenize=False, add_generation_prompt=True), return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
                with torch.no_grad():
                    _so = train_model.generate(**_si, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                _sr = tokenizer.decode(_so[0][_si['input_ids'].shape[1]:], skip_special_tokens=True)
                answer = extract_answer(_sr)
                soft_conf = get_soft_confidence(_so, _si['input_ids'].shape[1], train_model, tokenizer)
                del _so
                if soft_conf is not None:
                    confidence = p_true_to_bin(soft_conf)
                else:
                    confidence = extract_confidence(_sr)
                skip_response = _sr
            else:
                answer, confidence = baseline['answer'], baseline['confidence']
                skip_response = baseline['response']
            is_correct = answers_match(answer, qa, dataset_type=dataset_type)
            conf_prob = CONF_TO_PROB.get(confidence, 0.5)
            brier = (conf_prob - float(is_correct)) ** 2
            with torch.no_grad():
                skip_p_true = get_discriminative_confidence(question, answer, skip_response, train_model, tokenizer, use_p_know=args.use_p_know, claimed_bin=confidence)
            brier_p_true = (skip_p_true - float(is_correct)) ** 2
            result = {
                'question': question[:200], 'ground_truth': ground_truth, 'domain': domain,
                'baseline_answer': baseline['answer'], 'baseline_confidence': baseline['confidence'], 'baseline_correct': baseline_correct,
                'ttt_answer': answer, 'ttt_confidence': confidence, 'ttt_correct': is_correct, 'ttt_p_true': skip_p_true,
                'brier_p_true': brier_p_true, 'training_losses': [], 'neighbor_p_trues': [skip_p_true], 'avg_p_true': skip_p_true, 'ttt_skipped': True
            }
            avg_loss = 0.0
            avg_p_true = skip_p_true
            neighbor_p_trues = [skip_p_true]
        
        results.append(result)
        
        global_correct += int(is_correct)
        global_brier_sum += brier
        domain_stats[domain]['correct'] += int(is_correct)
        domain_stats[domain]['total'] += 1
        domain_stats[domain]['brier_sum'] += brier
        
        n = i + 1
        global_acc = global_correct / n
        global_brier = global_brier_sum / n
        
        if n >= 5:
            global_ece = compute_calibration_metrics(results, prefix="ttt")['ece']
        else:
            global_ece = 0.0
        
        learning_curve['step'].append(n)
        learning_curve['global_acc'].append(global_acc)
        learning_curve['global_brier'].append(global_brier)
        learning_curve['global_ece'].append(global_ece)
        learning_curve['avg_loss'].append(avg_loss)
        learning_curve['avg_p_true'].append(avg_p_true)
        
        if (i + 1) % 50 == 0:
            ckpt_suffix = args.run_name if args.run_name else args.mode
            with open(f'results/continual_ttt_{ckpt_suffix}_checkpoint.json', 'w') as f:
                json.dump(results, f, default=str)
            print(f"  [Checkpoint saved: {i+1} questions]")
        
        if (i + 1) % 25 == 0:
            print("\n" + "="*70)
            print(f"SAMPLE GENERATION @ Step {i+1} [{domain}]")
            print("="*70)
            q_preview = question[:150] + "..." if len(question) > 150 else question
            print(f"Q: {q_preview}")
            print(f"Ground Truth: {ground_truth}")
            print("-"*70)
            mark = "OK" if is_correct else "WRONG"
            print(f"TTT: {result['ttt_answer']} (conf: {result['ttt_confidence']}) {mark}")
            if not args.baseline_only:
                print(f"Avg P(True): {avg_p_true:.3f}")
            print("="*70 + "\n")
            
            if use_wandb:
                sample_table = wandb.Table(columns=[
                    "step", "domain", "question", "ground_truth",
                    "baseline_answer", "baseline_conf", "baseline_correct",
                    "ttt_answer", "ttt_conf", "ttt_p_true", "ttt_correct"
                ])
                sample_table.add_data(
                    n,
                    domain,
                    question,
                    ground_truth,
                    baseline.get('answer', ''),
                    baseline.get('confidence', ''),
                    baseline_correct,
                    result['ttt_answer'],
                    result['ttt_confidence'],
                    result.get('ttt_p_true', avg_p_true),
                    is_correct
                )
                wandb.log({"samples": sample_table}, step=n, commit=False)
        
        if use_wandb:
            log_dict = {
                "step": n,
                "ttt/brier": brier,
                "ttt/correct": int(is_correct),
                "ttt/confidence": result['ttt_confidence'],
                "ttt/p_true": result.get('ttt_p_true', avg_p_true),
                "ttt/rolling_acc": global_acc,
                "ttt/rolling_brier": global_brier,
                "ttt/rolling_ece": global_ece,
                "training/avg_loss": avg_loss,
                "discriminative/avg_p_true": avg_p_true,
                f"domain/{domain}/acc": domain_stats[domain]['correct'] / domain_stats[domain]['total'],
                f"domain/{domain}/brier": domain_stats[domain]['brier_sum'] / domain_stats[domain]['total'],
                f"domain/{domain}/correct": int(is_correct),
            }
            if 'brier_p_true' in result:
                log_dict["ttt/brier_p_true"] = result['brier_p_true']
            if neighbor_p_trues:
                log_dict["discriminative/min_p_true"] = min(neighbor_p_trues)
                log_dict["discriminative/max_p_true"] = max(neighbor_p_trues)
            if args.use_ph_gate and ph_gate is not None:
                log_dict["gate/ttt_active"] = int(should_do_ttt)
                log_dict["gate/ttt_remaining"] = ph_gate.ttt_remaining
                log_dict["gate/total_triggers"] = ph_gate.total_triggers
                log_dict["gate/ema_entropy"] = ph_gate.ema if ph_gate.ema else 0.0
                log_dict["gate/ttt_skipped"] = int(result.get('ttt_skipped', False))
                log_dict["gate/fresh_trigger"] = int(ph_gate.fresh_trigger)
                log_dict["gate/cooldown_remaining"] = ph_gate.cooldown_remaining
            wandb.log(log_dict, step=n)
        
        # Skip domain summaries in interleaved mode
        next_domain = questions[i + 1]['domain'] if i + 1 < len(questions) else None
        if next_domain != domain and args.mode != 'interleaved':
            domain_results_so_far = [r for r in results if r.get('domain') == domain]
            if domain_results_so_far:
                domain_cal = compute_calibration_metrics(domain_results_so_far, prefix="ttt")
                d_acc = domain_stats[domain]['correct'] / domain_stats[domain]['total']
                d_brier = domain_stats[domain]['brier_sum'] / domain_stats[domain]['total']
                d_ece = domain_cal['ece']
                d_auroc = domain_cal['auroc']
                
                print("\n" + "=" * 70)
                print(f"DOMAIN COMPLETE: {DOMAIN_CONFIG[domain]['name']}")
                print("=" * 70)
                print(f"  Accuracy: {d_acc:.1%} ({domain_stats[domain]['correct']}/{domain_stats[domain]['total']})")
                print(f"  Brier:    {d_brier:.4f}")
                print(f"  ECE:      {d_ece:.4f}")
                print(f"  AUROC:    {d_auroc:.4f}" if d_auroc else "  AUROC:    N/A")
                print("=" * 70 + "\n")
                
                if use_wandb:
                    wandb.log({
                        f"domain_final/{domain}/acc": d_acc,
                        f"domain_final/{domain}/brier": d_brier,
                        f"domain_final/{domain}/ece": d_ece,
                        f"domain_final/{domain}/auroc": d_auroc,
                    }, step=n, commit=False)
        
        if (i + 1) % 10 == 0:
            print(f"\n[{i+1}/{len(questions)}] Global: Acc={global_acc:.1%}  Brier={global_brier:.4f}  ECE={global_ece:.4f}")
            for d in set([qa['domain'] for qa in questions[:i+1]]):
                if d in domain_stats and domain_stats[d]['total'] > 0:
                    d_acc = domain_stats[d]['correct'] / domain_stats[d]['total']
                    d_brier = domain_stats[d]['brier_sum'] / domain_stats[d]['total']
                    print(f"  {DOMAIN_CONFIG[d]['name']}: Acc={d_acc:.1%}  Brier={d_brier:.4f}")
            if not args.baseline_only:
                print(f"  Avg P(True): {avg_p_true:.3f}")
    
    # Final metrics
    ttt_metrics = compute_calibration_metrics(results, prefix="ttt")
    
    domain_metrics = {}
    unique_domains = list(set([r['domain'] for r in results]))
    for domain in unique_domains:
        domain_results = [r for r in results if r['domain'] == domain]
        if domain_results:
            domain_cal = compute_calibration_metrics(domain_results, prefix="ttt")
            domain_brier = domain_stats[domain]['brier_sum'] / domain_stats[domain]['total']
            domain_metrics[domain] = {
                'acc': domain_stats[domain]['correct'] / domain_stats[domain]['total'],
                'brier': domain_brier,
                'ece': domain_cal['ece'],
                'auroc': domain_cal['auroc'],
                'count': domain_stats[domain]['total']
            }
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Overall Accuracy: {global_correct}/{len(questions)} = {global_acc:.1%}")
    print(f"Overall Brier: {global_brier:.4f}")
    print(f"Overall ECE: {ttt_metrics['ece']:.4f}")
    if ttt_metrics['auroc']:
        print(f"Overall AUROC: {ttt_metrics['auroc']:.4f}")
    
    print("\nPer-Domain Results (Acc / Brier / ECE / AUROC):")
    for domain in unique_domains:
        if domain in domain_metrics:
            m = domain_metrics[domain]
            auroc_str = f"{m['auroc']:.4f}" if m['auroc'] else "N/A"
            print(f"  {DOMAIN_CONFIG[domain]['name']:12} Acc={m['acc']:.1%}  Brier={m['brier']:.4f}  ECE={m['ece']:.4f}  AUROC={auroc_str}  (n={m['count']})")
    
    print(f"\nPer-bin accuracy:")
    for b, stats in ttt_metrics['bin_stats'].items():
        if stats['count'] > 0:
            acc_str = f"{stats['accuracy']:.1%}" if stats['accuracy'] is not None else "N/A"
            print(f"  Bin{b} (expect {stats['expected']:.0%}): {acc_str} ({stats['count']} samples)")
    
    print(f"\nConfidence distribution: {ttt_metrics['conf_distribution']}")
    
    if args.use_ph_gate and ph_gate is not None:
        ttt_trained = sum(1 for r in results if not r.get('ttt_skipped', False))
        ttt_skipped = sum(1 for r in results if r.get('ttt_skipped', False))
        print(f"\nPH Gate Summary:")
        print(f"  Total triggers: {ph_gate.total_triggers}")
        print(f"  TTT rounds: {ttt_trained} ({100*ttt_trained/len(results):.1f}%)")
        print(f"  Skipped: {ttt_skipped} ({100*ttt_skipped/len(results):.1f}%)")
    
    all_p_trues = [p for r in results for p in r.get('neighbor_p_trues', [])]
    if all_p_trues:
        print(f"\nP(True) Distribution:")
        print(f"  Mean: {np.mean(all_p_trues):.3f}")
        print(f"  Median: {np.median(all_p_trues):.3f}")
        print(f"  Std: {np.std(all_p_trues):.3f}")
        print(f"  Range: [{min(all_p_trues):.3f}, {max(all_p_trues):.3f}]")
    
    if use_wandb:
        log_dict = {
            "final/acc": global_acc,
            "final/brier": global_brier,
            "final/ece": ttt_metrics['ece'],
        }
        if ttt_metrics['auroc']:
            log_dict["final/auroc"] = ttt_metrics['auroc']
        
        for domain in unique_domains:
            if domain in domain_metrics:
                m = domain_metrics[domain]
                log_dict[f"final/{domain}_acc"] = m['acc']
                log_dict[f"final/{domain}_brier"] = m['brier']
                log_dict[f"final/{domain}_ece"] = m['ece']
                if m['auroc']:
                    log_dict[f"final/{domain}_auroc"] = m['auroc']
        
        if args.use_ph_gate and ph_gate is not None:
            ttt_trained = sum(1 for r in results if not r.get('ttt_skipped', False))
            log_dict["final/gate_triggers"] = ph_gate.total_triggers
            log_dict["final/ttt_trained_pct"] = ttt_trained / len(results)
        
        wandb.log(log_dict, step=len(questions))
        try:
            wandb.finish(quiet=True)
        except Exception:
            pass
    
    name_suffix = args.run_name if args.run_name else args.mode
    output_file = f"results/continual_ttt_{name_suffix}_results.json"
    save_data = {
        'config': vars(args),
        'domain_order': domain_order,
        'final_accuracy': global_acc,
        'final_brier': global_brier,
        'final_ece': ttt_metrics['ece'],
        'final_auroc': ttt_metrics['auroc'],
        'domain_metrics': domain_metrics,
        'domain_stats': {k: dict(v) for k, v in domain_stats.items()},
        'results': results
    }
    if args.use_ph_gate and ph_gate is not None:
        ttt_trained = sum(1 for r in results if not r.get('ttt_skipped', False))
        ttt_skipped = sum(1 for r in results if r.get('ttt_skipped', False))
        save_data['gate_stats'] = {
            'total_triggers': ph_gate.total_triggers,
            'ttt_trained': ttt_trained,
            'ttt_skipped': ttt_skipped,
            'ttt_trained_pct': ttt_trained / len(results)
        }
    with open(output_file, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {output_file}")
    
    return results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    run_continual_experiment()