#!/usr/bin/env python3
"""
Continual Test-Time Training with Discriminative Calibration
=============================================================
Same as run_ttt_last_layers.py but across multiple domains for continual learning.

Benchmark Modes:
  - sequential: GSM8K → MMLU → ARC → TruthfulQA (domain shift)
  - interleaved: Mixed questions from all domains
  - return: Domain A → Domain B → Domain A (forgetting test)

Run with: python run_continual_ttt.py --mode sequential --questions_per_domain 100
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
parser.add_argument('--n_neighbors', type=int, default=3, help='Number of neighborhood questions')
parser.add_argument('--n_epochs', type=int, default=3, help='Training epochs per question')
parser.add_argument('--k_distractors', type=int, default=4, help='Number of distractors for P(True) normalization')
parser.add_argument('--use_raw_ptrue', action='store_true', help='Use raw P(True) without normalization for target bin')
parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
parser.add_argument('--lora_r', type=int, default=8, help='LoRA rank')
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
parser.add_argument('--norm_temperature', type=float, default=0.7,
                    help='Temperature (<1 sharpen) when normalizing P(True) over answer+distractors')
parser.add_argument('--correction_every', type=int, default=0,
                    help='Every N questions, run a sparse supervised correction (0=off)')
parser.add_argument('--correction_lr', type=float, default=1e-5,
                    help='Learning rate for sparse corrections')
parser.add_argument('--correction_steps', type=int, default=1,
                    help='Steps for each correction application')
parser.add_argument('--correction_bin_high', type=int, default=4,
                    help='Target bin when the model is correct (anchor high confidence)')
parser.add_argument('--correction_bin_low', type=int, default=0,
                    help='Target bin when the model is incorrect (anchor low confidence)')
parser.add_argument('--seed', type=int, default=42, help='Random seed')
parser.add_argument('--num_bins', type=int, default=10, help='Number of confidence bins (rolling quantile edges)')
parser.add_argument('--bin_window', type=int, default=500, help='Rolling window size for bin quantiles')
parser.add_argument('--bin_min_count', type=int, default=100, help='Minimum samples before using quantile edges')
# PH-gated TTT arguments
parser.add_argument('--use_ph_gate', action='store_true', help='Enable PH-gated TTT (only train on detected drift)')
parser.add_argument('--ph_delta', type=float, default=0.05, help='PH tolerance parameter')
parser.add_argument('--ph_threshold', type=float, default=4.0, help='PH trigger threshold')
parser.add_argument('--ema_alpha', type=float, default=0.05, help='EMA smoothing factor (lower=smoother)')
parser.add_argument('--ttt_burst', type=int, default=20, help='Number of TTT rounds per trigger')
parser.add_argument('--ttt_warmup', type=int, default=50, help='Force TTT for first N questions (warmup)')
args = parser.parse_args()

# Handle accumulation flags
if args.no_accumulation:
    args.accumulation = False
elif not args.accumulation and not args.no_accumulation:
    args.accumulation = True  # Default to accumulation

# SET GPU BEFORE ANY IMPORTS
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

import torch
import torch.nn.functional as F
import numpy as np
import json
import re
import random
from datetime import datetime
from collections import defaultdict, deque
from functools import partial
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
import wandb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

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
    # TruthfulQA MC default
    'sequential': ['gsm8k', 'mmlu', 'arc', 'truthfulqa'],
    'reversed': ['truthfulqa', 'arc', 'mmlu', 'gsm8k'],
    # TruthfulQA generation variant
    'sequential_gen': ['gsm8k', 'mmlu', 'arc', 'truthfulqa_gen'],
    'reversed_gen': ['truthfulqa_gen', 'arc', 'mmlu', 'gsm8k'],

    'sequential_hard': ['gsm8k', 'mmlu', 'arc', 'truthfulqa', 'hle'],
    'interleaved': ['gsm8k', 'mmlu', 'arc', 'truthfulqa'],
    'return_gsm8k': ['gsm8k', 'mmlu', 'gsm8k'],
    'return_mmlu': ['mmlu', 'arc', 'mmlu'],
    # Shuffled orders to test domain order effects
    'truthful_first': ['truthfulqa', 'gsm8k', 'mmlu', 'arc'],
    'mmlu_first': ['mmlu', 'arc', 'truthfulqa', 'gsm8k'],
    'arc_first': ['arc', 'truthfulqa', 'gsm8k', 'mmlu'],
}

LATE_LAYER_START = 24  # Tune layers 24-31
NUM_LAYERS = 32

NUM_BINS = args.num_bins
CONF_TO_PROB = {i: (i + 0.5) / NUM_BINS for i in range(NUM_BINS)}

create_qa_prompt = partial(_create_qa_prompt, num_bins=NUM_BINS)
extract_confidence = partial(_extract_confidence, num_bins=NUM_BINS)

print("=" * 70)
print("CONTINUAL TTT WITH DISCRIMINATIVE CALIBRATION - LATE LAYERS (24-31)")
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
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    low_cpu_mem_usage=True
).to(DEVICE)
base_model.eval()
print(f"Base model loaded to: {next(base_model.parameters()).device}")
print(f"Model has {NUM_LAYERS} layers, will target late layers: {LATE_LAYER_START}-{NUM_LAYERS-1}")

if NEIGHBOR_MODEL_NAME == MODEL_NAME:
    # Important VRAM optimization: don't load the same model twice.
    # Many runs set --neighbor_model == --model_name (default), so reusing saves a lot of memory.
    print(f"Neighbor model == base model ({NEIGHBOR_MODEL_NAME}); reusing base model/tokenizer for neighbor generation.")
    neighbor_tokenizer = tokenizer
    neighbor_model = base_model
else:
    print(f"Loading {NEIGHBOR_MODEL_NAME}...")
    neighbor_tokenizer = AutoTokenizer.from_pretrained(NEIGHBOR_MODEL_NAME)
    neighbor_tokenizer.pad_token = neighbor_tokenizer.eos_token
    neighbor_tokenizer.padding_side = "left"

    neighbor_model = AutoModelForCausalLM.from_pretrained(
        NEIGHBOR_MODEL_NAME,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    ).to(DEVICE)
    neighbor_model.eval()
    print(f"Neighbor model loaded to: {next(neighbor_model.parameters()).device}")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_late_layers_target_modules(start_layer, end_layer):
    """Generate target_modules for layers [start_layer, end_layer)."""
    targets = []
    for layer_idx in range(start_layer, end_layer):
        targets.append(f"model.layers.{layer_idx}.self_attn.q_proj")
        targets.append(f"model.layers.{layer_idx}.self_attn.v_proj")
    return targets


class RollingBinMapper:
    """Maintain rolling quantile bin edges for dynamic binning."""

    def __init__(self, num_bins: int, window_size: int = 500, min_count: int = 100):
        self.num_bins = num_bins
        self.window = deque(maxlen=window_size)
        self.min_count = min_count
        self.edges = np.linspace(0.0, 1.0, num_bins + 1)

    def add(self, value: float):
        self.window.append(float(value))
        if len(self.window) >= self.min_count:
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


bin_mapper = RollingBinMapper(NUM_BINS, args.bin_window, args.bin_min_count)


class PHGate:
    """Page-Hinkley gated TTT trigger with fixed burst."""
    
    def __init__(self, ema_alpha=0.05, ph_delta=0.05, ph_threshold=4.0, 
                 ttt_burst=20, warmup=50):
        self.ema_alpha = ema_alpha
        self.ph_delta = ph_delta
        self.ph_threshold = ph_threshold
        self.ttt_burst = ttt_burst
        self.warmup = warmup
        
        self.ema = None
        self.data = []
        self.segment_start = 0
        self.m_up = self.m_up_min = 0.0
        self.m_down = self.m_down_min = 0.0
        
        self.ttt_remaining = 0
        self.total_triggers = 0
        self.t = 0
    
    def update(self, entropy: float) -> bool:
        """Update with new entropy value. Returns True if TTT should run."""
        self.t += 1
        
        # Update EMA
        if self.ema is None:
            self.ema = entropy
        else:
            self.ema = self.ema_alpha * entropy + (1 - self.ema_alpha) * self.ema
        
        self.data.append(self.ema)
        
        # Warmup: always TTT
        if self.t <= self.warmup:
            return True
        
        # Bidirectional PH test
        segment_data = self.data[self.segment_start:]
        mean_t = np.mean(segment_data)
        
        # Upward shift
        dev_up = self.ema - mean_t - self.ph_delta
        self.m_up += dev_up
        if len(segment_data) == 1:
            self.m_up_min = self.m_up
        else:
            self.m_up_min = min(self.m_up_min, self.m_up)
        ph_u = self.m_up - self.m_up_min
        
        # Downward shift
        dev_down = mean_t - self.ema - self.ph_delta
        self.m_down += dev_down
        if len(segment_data) == 1:
            self.m_down_min = self.m_down
        else:
            self.m_down_min = min(self.m_down_min, self.m_down)
        ph_d = self.m_down - self.m_down_min
        
        # Check trigger
        if ph_u > self.ph_threshold or ph_d > self.ph_threshold:
            self.ttt_remaining = self.ttt_burst
            self.total_triggers += 1
            # Reset PH for next segment
            self.segment_start = len(self.data)
            self.m_up = self.m_down = 0.0
            self.m_up_min = self.m_down_min = 0.0
        
        # Decide
        if self.ttt_remaining > 0:
            self.ttt_remaining -= 1
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
    
    # Single prompt to generate all neighbors at once
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
    
    # Simple parse: each line that starts with a digit is a question
    for line in response.split('\n'):
        line = line.strip()
        if line and line[0].isdigit():
            # Remove "1." or "1)" prefix
            q = line.lstrip('0123456789').lstrip('.):- ').strip()
            if len(q) > 15:
                all_questions.append({'question': q, 'difficulty': 'similar'})
    
    # Fallback to original
    while len(all_questions) < n_neighbors:
        all_questions.append({'question': question, 'difficulty': 'similar'})
    
    return all_questions[:n_neighbors]


def _is_too_similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Check if two answers are too similar (character-level)."""
    a_clean = normalize_answer(a)
    b_clean = normalize_answer(b)
    if not a_clean or not b_clean:
        return True
    # Exact match
    if a_clean == b_clean:
        return True
    # Check if one is substring of other
    if a_clean in b_clean or b_clean in a_clean:
        return True
    # Character overlap ratio
    a_set = set(a_clean)
    b_set = set(b_clean)
    if not a_set or not b_set:
        return True
    overlap = len(a_set & b_set) / max(len(a_set), len(b_set))
    return overlap > threshold


def _extract_mc_options(question_text: str) -> dict:
    """Extract multiple-choice options from question text.
    
    Returns dict like {'A': 'option text', 'B': 'option text', ...}
    """
    options = {}
    for line in question_text.splitlines():
        line = line.strip()
        # Match patterns like "A. text", "A) text", "A: text"
        m = re.match(r'^([A-D])[\.\)\:]?\s*(.+)', line)
        if m:
            letter = m.group(1).upper()
            text = m.group(2).strip()
            options[letter] = text
    return options


def _is_mc_answer(answer: str) -> str:
    """Check if answer is a single MC letter (A/B/C/D). Returns the letter or None."""
    answer_clean = answer.strip().upper().rstrip('.')
    if answer_clean in ['A', 'B', 'C', 'D']:
        return answer_clean
    return None


def generate_distractors(question, base_answer, model, tokenizer, k=2):
    """Generate k plausible alternative answers for a question.
    
    For MCQ questions: uses the other options (A/B/C/D) as distractors.
    For open-ended: generates plausible alternatives via the model.
    """
    base_answer_str = str(base_answer).strip()
    
    # Check if this is an MCQ answer (A/B/C/D)
    mc_letter = _is_mc_answer(base_answer_str)
    if mc_letter:
        # Extract options from the question
        mc_options = _extract_mc_options(question)
        if mc_options:
            # Use other options as distractors (full text for stronger contrast)
            distractors = []
            for letter in ['A', 'B', 'C', 'D']:
                if letter != mc_letter and letter in mc_options:
                    distractors.append(f"{letter}. {mc_options[letter]}")
                    if len(distractors) >= k:
                        break
            if distractors:
                return distractors[:k]
    
    # For non-MCQ or if MC extraction failed: generate distractors
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
        # Strip numeric prefixes like "1." or "1)"
        line = re.sub(r"^\d+[\.)]\s*", "", line).strip()
        if not line:
            continue
        
        # Drop if identical or too similar to base answer
        if _is_too_similar(line, base_answer_str):
            continue
        
        # Drop if too similar to already collected distractors
        if any(_is_too_similar(line, c) for c in candidates):
            continue
        
        # Keep concise answers only (for non-numeric)
        if len(line.split()) > 6 and not is_numeric:
            continue
        
        if is_numeric:
            # For numeric answers, require that the line starts with a number
            if not re.match(r"^-?\d", line):
                continue
        
        candidates.append(line)
        if len(candidates) >= k:
            break

    # No fallbacks - return only actual distractors we generated
    return candidates[:k]


def get_discriminative_confidence(question, answer, reasoning, model, tokenizer, use_p_know=False, claimed_bin=None, candidate_list=None):
    """Get P(True) or P(Know) for the model's judgment.
    
    If use_p_know=True and claimed_bin is provided:
      - Asks "Are you at least X% confident?" where X = claimed percentage
      - Returns: claimed_confidence * P(Know)
      - This scales down confidence when model doesn't believe its own claim
    """
    bin_to_pct = {i: (i + 0.5) / NUM_BINS for i in range(NUM_BINS)}
    
    # Optional context of other candidate answers (for contrastive awareness)
    candidates_section = ""
    if candidate_list:
        candidates_lines = []
        for idx, cand in enumerate(candidate_list, 1):
            candidates_lines.append(f"{idx}. {cand}")
        candidates_section = "Here are some brainstormed answers:\n" + "\n".join(candidates_lines) + "\n\n"
    
    if use_p_know:
        # P(Know): Check if model believes its claimed confidence
        if claimed_bin is not None:
            claimed_pct = bin_to_pct.get(claimed_bin, 0.50)
            pct_display = int(claimed_pct * 100)
            verification_prompt = f"""Question: {question}

Are you at least {pct_display}% confident that you know the correct answer to this question?
Please respond with only "True" or "False"."""
        else:
            # Fallback to 100% if no claimed_bin provided
            claimed_pct = 1.0
            verification_prompt = f"""Question: {question}

Are you 100% confident that you know the correct answer to this question?
Please respond with only "True" or "False"."""
    else:
        # P(True): "Is this specific answer correct?" (answer verification)
        claimed_pct = None  # Not used for P(True)
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
        # Binary gate with 70% threshold
        # If P(Know) > 0.7: model strongly believes its claim, keep it
        # If P(Know) <= 0.7: model doubts, scale down
        if p_know > 0.7:
            return claimed_pct  # Keep claimed confidence
        else:
            scaled_confidence = claimed_pct * p_know
            return scaled_confidence
    else:
        # P(True) mode: return raw probability
        return p_know


def p_true_to_bin(p_true):
    """Map P(True) to dynamic confidence bin using rolling quantile edges."""
    return bin_mapper.to_bin(p_true)


def get_bin_digit_tokens(tokenizer):
    """Get token IDs for each confidence bin digit."""
    return {i: tokenizer.encode(str(i), add_special_tokens=False)[-1] for i in range(NUM_BINS)}


def normalize_p_true_sharpened(p_scores, temperature=1.0):
    """Normalize P(True) across candidates with optional sharpening."""
    scores = torch.tensor(p_scores, dtype=torch.float32)
    scores = scores.clamp(min=1e-6)
    temp = max(temperature, 1e-6)
    weights = torch.softmax(torch.log(scores) / temp, dim=0)
    return float(weights[0].item())


def create_confidence_mask(token_ids, tokenizer):
    """Create mask that ONLY targets the confidence bin digit."""
    mask = torch.zeros_like(token_ids, dtype=torch.float)
    
    response_text = tokenizer.decode(token_ids[0], skip_special_tokens=True)
    
    conf_match = response_text.lower().find("confidence:")
    if conf_match == -1:
        return mask
    
    conf_section = response_text[conf_match:conf_match+30]
    bin_match = re.search(r'bin(\d+)', conf_section, re.IGNORECASE)
    if not bin_match:
        return mask
    
    digit_char_pos = conf_match + bin_match.end() - 1
    current_char = 0
    for i, tid in enumerate(token_ids[0]):
        token_text = tokenizer.decode([tid.item()], skip_special_tokens=True)
        if current_char <= digit_char_pos < current_char + len(token_text):
            mask[0, i] = 1.0
            break
        current_char += len(token_text)
    
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


def get_baseline_answer(question, model, tokenizer):
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
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del outputs
    
    return {
        'answer': extract_answer(response),
        'confidence': extract_confidence(response),
        'response': response
    }


# =============================================================================
# DISCRIMINATIVE TTT TRAINING
# =============================================================================

def parse_mc_options(question_text):
    """Extract multiple-choice options from question text."""
    opts = []
    for line in question_text.splitlines():
        line = line.strip()
        m = re.match(r'^([A-D])[\.\)]\s*(.+)', line)
        if m:
            opts.append(m.group(2).strip())
    return opts


def best_option_first(answer_text, options):
    """Reorder options so that the best match to the answer is first."""
    norm_ans = normalize_answer(answer_text)
    best_idx = 0
    best_score = -1
    for i, opt in enumerate(options):
        norm_opt = normalize_answer(opt)
        # token overlap
        toks_ans = set(norm_ans.split())
        toks_opt = set(norm_opt.split())
        if toks_ans and toks_opt:
            overlap = len(toks_ans & toks_opt) / max(len(toks_ans), len(toks_opt))
        else:
            overlap = 0
        if overlap > best_score:
            best_score = overlap
            best_idx = i
    if best_idx == 0:
        return options
    return [options[best_idx]] + [opt for i, opt in enumerate(options) if i != best_idx]


def train_single_question_discriminative(question, train_model, optimizer, tokenizer, n_neighbors=5, n_epochs=3, dataset_type="auto"):
    """Train on a single question using discriminative P(True) as pseudo-label."""
    
    bin_digit_tokens = get_bin_digit_tokens(tokenizer)
    pknow_values = []
    
    train_model.eval()
    with torch.no_grad():
        neighbors = generate_neighborhood_questions(question, neighbor_model, neighbor_tokenizer, n_neighbors)
    
    if not neighbors:
        neighbors = [{'question': question, 'difficulty': 'similar'}]
    
    neighbor_data = []
    for n in neighbors:
        q = n['question'] if isinstance(n, dict) else n
        difficulty = n.get('difficulty', 'similar') if isinstance(n, dict) else 'similar'
        
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
        claimed_conf = extract_confidence(response)  # Get claimed confidence bin
        # Epistemic component: P(Know) for claimed bin
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
        
        # Generate candidates and compute normalized P(True)
        # If MCQ, use option text; else generated distractors
        mc_opts = _extract_mc_options(question) if dataset_type in ['mmlu', 'arc', 'truthfulqa_mc'] else {}
        if mc_opts:
            # Ordered by letter
            ordered = [f"{ltr}. {mc_opts[ltr]}" for ltr in sorted(mc_opts.keys())]
            candidates = ordered
            # Map model answer letter to full text if possible
            ans_letter = answer.strip().upper().rstrip('.')
            answer_full = mc_opts.get(ans_letter, answer)
            candidates.insert(0, f"{ans_letter}. {answer_full}")
        else:
            distractors = generate_distractors(q, answer, train_model, tokenizer, k=args.k_distractors)
            candidates = [answer] + distractors
        
        # Fallback flag: if no distractors, normalization is trivial → use raw P(True)
        no_distractors = len(candidates) <= 1
        
        p_scores = []
        with torch.no_grad():
            for cand in candidates:
                p_cand = get_discriminative_confidence(
                    q,
                    cand,
                    response,  # reuse reasoning; question/answer checked in prompt
                    train_model,
                    tokenizer,
                    use_p_know=False,  # force P(True) for normalization
                    claimed_bin=None,
                    candidate_list=candidates,
                )
                p_scores.append(p_cand)
        p_true_raw = p_scores[0]
        
        # If no distractors, normalization is trivial (would be 1.0), fall back to raw
        if no_distractors:
            p_true_norm = p_true_raw  # Fallback: use raw when no contrast
        else:
            p_true_norm = normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
        
        p_know_mean = float(np.mean(pknow_values)) if pknow_values else 0.5
        p_fused = p_true_norm * p_know_mean
        
        if args.use_raw_ptrue or no_distractors:
            p_for_bin = p_true_raw  # raw P(True)
        elif args.use_fused_confidence:
            p_for_bin = p_fused     # fused P(True) * P(Know)
        else:
            p_for_bin = p_true_norm  # normalized P(True)

        bin_mapper.add(p_for_bin)
        target_bin = p_true_to_bin(p_for_bin)
        
        neighbor_data.append({
            'question': q,
            'difficulty': difficulty,
            'answer': answer,
            'response': response,
            'p_true_raw': p_true_raw,
            'p_true_norm': p_true_norm,
            'p_know': p_know,
            'p_know_mean': p_know_mean,
            'p_fused': p_fused,
            'target_bin': target_bin
        })
    
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
            
            # Try greedy first, then retry with sampling if format fails
            max_retries = 3
            outputs = None
            conf_mask = None
            
            for retry in range(max_retries):
                with torch.no_grad():
                    if retry == 0:
                        # First try: greedy decoding
                        outputs = train_model.generate(**inputs, **generation_kwargs)
                    else:
                        # Retry with sampling
                        outputs = train_model.generate(
                            **inputs,
                            max_new_tokens=200,
                            do_sample=True,
                            temperature=0.7,
                            pad_token_id=tokenizer.eos_token_id
                        )
                
                response_tokens = outputs[:, inputs['input_ids'].shape[1]:]
                conf_mask = create_confidence_mask(response_tokens, tokenizer)
                
                if conf_mask.sum() > 0:
                    break  # Success
                
                del outputs
                outputs = None
            
            if outputs is None or conf_mask.sum() == 0:
                # All retries failed
                if outputs is not None:
                    resp_text = tokenizer.decode(response_tokens[0], skip_special_tokens=True)
                    conf_idx = resp_text.lower().find("confidence:")
                    if conf_idx != -1:
                        conf_section = resp_text[conf_idx:conf_idx+30]
                        print(f"  [MASK FAIL after {max_retries} retries] conf_section='{conf_section}'")
                    else:
                        print(f"  [MASK FAIL after {max_retries} retries] No 'Confidence:' (len={len(resp_text)})")
                    del outputs
                continue
            
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
        
        training_losses.append(np.mean(epoch_losses) if epoch_losses else 0.0)
    
    train_model.eval()
    prompt = create_qa_prompt(question)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
    
    with torch.no_grad():
        outputs = train_model.generate(**inputs, **generation_kwargs)
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    del outputs
    
    answer = extract_answer(response)
    claimed_conf = extract_confidence(response)
    
    # Compute P(True) or P(Know) for the final answer
    with torch.no_grad():
        final_p_true = get_discriminative_confidence(question, answer, response, train_model, tokenizer, 
                                                     use_p_know=args.use_p_know, claimed_bin=claimed_conf)
    
    neighbor_p_know_mean = float(np.mean([n.get('p_know', 0.5) for n in neighbor_data])) if neighbor_data else 0.5
    neighbor_p_fused_mean = float(np.mean([n.get('p_fused', n.get('p_true_norm', 0.5)) for n in neighbor_data])) if neighbor_data else 0.5
    
    return {
        'answer': answer,
        'confidence': claimed_conf,
        'response': response,
        'neighbors': neighbor_data,
        'training_losses': training_losses,
        'p_true': final_p_true,  # P(True) for final answer
        'neighbor_p_know_mean': neighbor_p_know_mean,
        'neighbor_p_fused_mean': neighbor_p_fused_mean
    }


# =============================================================================
# MAIN EXPERIMENT
# =============================================================================

def run_continual_experiment():
    """Run continual TTT experiment across multiple domains."""
    
    bin_digit_tokens = get_bin_digit_tokens(tokenizer)
    
    # Determine domain order
    if args.mode == 'custom' and args.domains:
        domain_order = args.domains
    else:
        domain_order = BENCHMARK_MODES[args.mode]
    
    no_accumulation = not args.accumulation
    use_wandb = not args.no_wandb
    
    print(f"\n{'='*70}")
    print("RUNNING CONTINUAL DISCRIMINATIVE TTT - LATE LAYERS (24-31)")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    print(f"Domain order: {' → '.join(domain_order)}")
    print(f"Questions per domain: {args.questions_per_domain}")
    print(f"Neighbors per question: {args.n_neighbors}")
    print(f"Epochs per question: {args.n_epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight accumulation: {'OFF' if no_accumulation else 'ON'}")
    print(f"Baseline only: {args.baseline_only}")
    print()
    
    # Load all datasets
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
    
    # Build question sequence
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
    
    # Initialize W&B
    if use_wandb:
        if args.run_name:
            run_name = args.run_name
        else:
            timestamp = datetime.now().strftime("%m%d_%H%M")
            accum_str = "noaccum" if no_accumulation else "accum"
            baseline_str = "BASELINE_" if args.baseline_only else ""
            run_name = f"{baseline_str}CONT_{args.mode}_{accum_str}_nb{args.n_neighbors}_{timestamp}"
        
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            settings=wandb.Settings(console="off"),
            config={
                "method": "continual_discriminative_late_layers",
                "mode": args.mode,
                "domains": domain_order,
                "questions_per_domain": args.questions_per_domain,
                "n_neighbors": args.n_neighbors,
                "n_epochs": args.n_epochs,
                "lr": args.lr,
                "model": MODEL_NAME,
                "neighbor_model": NEIGHBOR_MODEL_NAME,
                "accumulation": not no_accumulation,
                "baseline_only": args.baseline_only,
                "target_layers": f"late ({LATE_LAYER_START}-{NUM_LAYERS-1})",
            }
        )
        print(f"W&B run: {run_name}")
    
    # Create LoRA adapter
    target_modules = get_late_layers_target_modules(LATE_LAYER_START, NUM_LAYERS)
    print(f"Target modules ({len(target_modules)}): {target_modules[:4]}...")
    
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.0,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    train_model = get_peft_model(base_model, lora_config)
    train_model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr) if not args.baseline_only else None
    
    # Tracking
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
    
    # Initialize PH gate for gated TTT
    ph_gate = None
    if args.use_ph_gate:
        ph_gate = PHGate(
            ema_alpha=args.ema_alpha,
            ph_delta=args.ph_delta,
            ph_threshold=args.ph_threshold,
            ttt_burst=args.ttt_burst,
            warmup=args.ttt_warmup
        )
        print(f"PH Gate enabled: burst={args.ttt_burst}, warmup={args.ttt_warmup}, "
              f"delta={args.ph_delta}, threshold={args.ph_threshold}, ema_alpha={args.ema_alpha}")
    
    # Main loop
    for i, qa in enumerate(tqdm(questions, desc=f"Continual TTT ({args.mode})")):
        question = qa['question']
        ground_truth = qa['ground_truth_answer']
        domain = qa['domain']
        dataset_type = qa['dataset_type']
        
        # Determine if TTT should run (PH gate check)
        should_do_ttt = True  # Default: always TTT
        if args.use_ph_gate and not args.baseline_only:
            with train_model.disable_adapter():
                entropy = compute_question_entropy(question, train_model, tokenizer)
            should_do_ttt = ph_gate.update(entropy)
        
        # Reset LoRA weights if no accumulation (only when doing TTT)
        # NOTE: Must reinit lora_A (Kaiming) and zero lora_B. If both are zeroed, gradients are 0!
        if no_accumulation and not args.baseline_only and should_do_ttt:
            with torch.no_grad():
                for name, param in train_model.named_parameters():
                    if 'lora_A' in name:
                        torch.nn.init.kaiming_uniform_(param, a=5**0.5)
                    elif 'lora_B' in name:
                        param.zero_()
            optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr)
        
        # Baseline answer
        with train_model.disable_adapter():
            baseline = get_baseline_answer(question, train_model, tokenizer)
        baseline_correct = answers_match(baseline['answer'], qa, dataset_type=dataset_type)
        
        if args.baseline_only:
            # Baseline only mode - also compute P(True) for comparison
            is_correct = baseline_correct

            # Compute P(True) or P(Know) for baseline answer (no training, just discriminative eval)
            with torch.no_grad():
                baseline_p_true = get_discriminative_confidence(
                    question, baseline['answer'], baseline['response'], train_model, tokenizer,
                    use_p_know=args.use_p_know, claimed_bin=baseline['confidence']
                )

            # Optionally override confidence bin with P(True)_norm (no training)
            if args.baseline_use_ptrue_norm:
                # Build candidates with distractors and compute normalized P(True)
                distractors = generate_distractors(question, baseline['answer'], train_model, tokenizer, k=args.k_distractors)
                candidates = [baseline['answer']] + distractors
                
                # Fallback flag: if no distractors, normalization is trivial → use raw P(True)
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
                
                # If no distractors, use raw P(True) as fallback
                if no_distractors:
                    p_true_norm = p_true_raw
                else:
                    p_soft = normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
                    distractor_mean = float(np.mean(p_scores[1:])) if len(p_scores) > 1 else 1e-6
                    p_ratio = p_true_raw / (p_true_raw + distractor_mean + 1e-8)
                    p_true_norm = max(p_soft, p_ratio)
                
                # Map to bin and use as reported confidence
                bin_mapper.add(p_true_norm)
                baseline_conf_bin = p_true_to_bin(p_true_norm)
                conf_prob = CONF_TO_PROB.get(baseline_conf_bin, 0.5)
                brier = (conf_prob - float(is_correct)) ** 2
                brier_p_true = (p_true_norm - float(is_correct)) ** 2  # Using p_true_norm for comparison
                # Override for logging/outputs
                baseline_conf_for_log = baseline_conf_bin
                neighbor_p_trues = [p_true_norm]
                avg_p_true = p_true_norm
            else:
                conf_prob = CONF_TO_PROB.get(baseline['confidence'], 0.5)
                brier = (conf_prob - float(is_correct)) ** 2
                brier_p_true = (baseline_p_true - float(is_correct)) ** 2  # Brier using P(True)
                baseline_conf_for_log = baseline['confidence']
                neighbor_p_trues = []
                avg_p_true = baseline_p_true
            
            result = {
                'question': question[:200],
                'ground_truth': ground_truth,
                'domain': domain,
                'baseline_answer': baseline['answer'],
                'baseline_confidence': baseline_conf_for_log,
                'baseline_p_true': baseline_p_true,  # Record P(True) for comparison
                'baseline_correct': is_correct,
                'ttt_answer': baseline['answer'],
                'ttt_confidence': baseline_conf_for_log,
                'ttt_correct': is_correct,
                'ttt_p_true': baseline_p_true,
                'training_losses': [],
                'neighbor_p_trues': neighbor_p_trues,
                'avg_p_true': avg_p_true,
                'brier_p_true': brier_p_true  # Track Brier with P(True)
            }
            avg_loss = 0.0
            # avg_p_true already set above
            # neighbor_p_trues already set above
            
        elif should_do_ttt:
            # TTT mode - run training
            baseline_correct = answers_match(baseline['answer'], qa, dataset_type=dataset_type)
            
            ttt_result = train_single_question_discriminative(
                question, train_model, optimizer, tokenizer,
                n_neighbors=args.n_neighbors, n_epochs=args.n_epochs, dataset_type=dataset_type
            )
            
            is_correct = answers_match(ttt_result['answer'], qa, dataset_type=dataset_type)
            
            # Use verbalized confidence for main Brier
            conf_prob = CONF_TO_PROB.get(ttt_result['confidence'], 0.5)
            brier = (conf_prob - float(is_correct)) ** 2
            
            if args.use_fused_confidence:
                neighbor_p_trues = [n.get('p_fused', n.get('p_true_norm', 0.5)) for n in ttt_result['neighbors']]
            else:
                neighbor_p_trues = [n.get('p_true_norm', 0.5) for n in ttt_result['neighbors']]
            avg_p_true = np.mean(neighbor_p_trues) if neighbor_p_trues else 0.5
            avg_loss = np.mean(ttt_result['training_losses']) if ttt_result['training_losses'] else 0.0
            
            brier_p_true = (ttt_result['p_true'] - float(is_correct)) ** 2  # Brier using P(True)
            
            result = {
                'question': question[:200],
                'ground_truth': ground_truth,
                'domain': domain,
                'baseline_answer': baseline['answer'],
                'baseline_confidence': baseline['confidence'],
                'baseline_correct': baseline_correct,
                'ttt_answer': ttt_result['answer'],
                'ttt_confidence': ttt_result['confidence'],  # Verbalized confidence bin
                'ttt_correct': is_correct,
                'ttt_p_true': ttt_result['p_true'],  # Store P(True) for final answer
                'ttt_p_know_mean': ttt_result.get('neighbor_p_know_mean', 0.5),
                'ttt_p_fused_mean': ttt_result.get('neighbor_p_fused_mean', avg_p_true),
                'brier_p_true': brier_p_true,  # Brier using P(True) for comparison
                'training_losses': ttt_result['training_losses'],
                'neighbor_p_trues': neighbor_p_trues,
                'avg_p_true': avg_p_true,
                'ttt_skipped': False
            }
        
        else:
            # Gated: Skip TTT, use model with adapter (no training)
            prompt = create_qa_prompt(question)
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
            del outputs
            
            answer = extract_answer(response)
            confidence = extract_confidence(response)
            is_correct = answers_match(answer, qa, dataset_type=dataset_type)
            
            conf_prob = CONF_TO_PROB.get(confidence, 0.5)
            brier = (conf_prob - float(is_correct)) ** 2
            
            result = {
                'question': question[:200],
                'ground_truth': ground_truth,
                'domain': domain,
                'baseline_answer': baseline['answer'],
                'baseline_confidence': baseline['confidence'],
                'baseline_correct': baseline_correct,
                'ttt_answer': answer,
                'ttt_confidence': confidence,
                'ttt_correct': is_correct,
                'training_losses': [],
                'neighbor_p_trues': [],
                'avg_p_true': 0.5,
                'ttt_skipped': True
            }
            avg_loss = 0.0
            avg_p_true = 0.5
            neighbor_p_trues = []
        
        # Optional sparse correction: use ground-truth correctness to anchor confidence bins
        if (args.correction_every > 0
                and not args.baseline_only
                and ((i + 1) % args.correction_every == 0)):
            target_bin = args.correction_bin_high if is_correct else args.correction_bin_low
            corr_opt = torch.optim.AdamW(train_model.parameters(), lr=args.correction_lr)
            # Rebuild prompt/inputs for this question
            prompt = create_qa_prompt(question)
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
            
            for _ in range(max(1, args.correction_steps)):
                with torch.no_grad():
                    outputs = train_model.generate(
                        **inputs,
                        max_new_tokens=200,
                        do_sample=False,
                        pad_token_id=tokenizer.eos_token_id
                    )
                response_tokens = outputs[:, inputs['input_ids'].shape[1]:]
                conf_mask = create_confidence_mask(response_tokens, tokenizer)
                if conf_mask.sum() == 0:
                    del outputs
                    break
                
                labels = outputs.clone()
                labels[:, :inputs['input_ids'].shape[1]] = -100
                response_labels = labels[:, inputs['input_ids'].shape[1]:]
                target_token_id = bin_digit_tokens[target_bin]
                for j in range(response_tokens.shape[1]):
                    if conf_mask[0, j] == 1:
                        response_labels[0, j] = target_token_id
                    else:
                        response_labels[0, j] = -100
                labels[:, inputs['input_ids'].shape[1]:] = response_labels
                
                corr_opt.zero_grad()
                model_output = train_model(
                    input_ids=outputs,
                    attention_mask=torch.ones_like(outputs),
                    labels=labels
                )
                if model_output.loss is not None and not torch.isnan(model_output.loss):
                    model_output.loss.backward()
                    torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
                    corr_opt.step()
                del outputs
        
        results.append(result)
        
        # Update stats
        global_correct += int(is_correct)
        global_brier_sum += brier
        domain_stats[domain]['correct'] += int(is_correct)
        domain_stats[domain]['total'] += 1
        domain_stats[domain]['brier_sum'] += brier
        
        n = i + 1
        global_acc = global_correct / n
        global_brier = global_brier_sum / n
        
        # Rolling ECE
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
        
        # Checkpoint every 50 questions
        if (i + 1) % 50 == 0:
            with open(f'continual_ttt_{args.mode}_checkpoint.json', 'w') as f:
                json.dump(results, f, default=str)
            print(f"  [Checkpoint saved: {i+1} questions]")
        
        # Sample generation every 25 questions
        if (i + 1) % 25 == 0:
            print("\n" + "="*70)
            print(f"SAMPLE GENERATION @ Step {i+1} [{domain}]")
            print("="*70)
            q_preview = question[:150] + "..." if len(question) > 150 else question
            print(f"Q: {q_preview}")
            print(f"Ground Truth: {ground_truth}")
            print("-"*70)
            mark = "✓" if is_correct else "✗"
            print(f"TTT: {result['ttt_answer']} (conf: {result['ttt_confidence']}) {mark}")
            if not args.baseline_only:
                print(f"Avg P(True): {avg_p_true:.3f}")
            print("="*70 + "\n")
            
            # Log sample to W&B (with baseline and TTT answers)
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
        
        # W&B logging
        if use_wandb:
            # Per-question metrics (not rolling)
            log_dict = {
                "step": n,
                "ttt/brier": brier,  # Per-question Brier (verbalized confidence)
                "ttt/correct": int(is_correct),
                "ttt/confidence": result['ttt_confidence'],
                "ttt/p_true": result.get('ttt_p_true', avg_p_true),  # P(True) for final answer
                # Rolling metrics
                "ttt/rolling_acc": global_acc,
                "ttt/rolling_brier": global_brier,
                "ttt/rolling_ece": global_ece,
                # Training
                "training/avg_loss": avg_loss,
                "discriminative/avg_p_true": avg_p_true,
                # Per-domain (within domain so far)
                f"domain/{domain}/acc": domain_stats[domain]['correct'] / domain_stats[domain]['total'],
                f"domain/{domain}/brier": domain_stats[domain]['brier_sum'] / domain_stats[domain]['total'],
                f"domain/{domain}/correct": int(is_correct),
            }
            # Log P(True)-based Brier if available
            if 'brier_p_true' in result:
                log_dict["ttt/brier_p_true"] = result['brier_p_true']
            if neighbor_p_trues:
                log_dict["discriminative/min_p_true"] = min(neighbor_p_trues)
                log_dict["discriminative/max_p_true"] = max(neighbor_p_trues)
            # Log PH gate metrics
            if args.use_ph_gate and ph_gate is not None:
                log_dict["gate/ttt_active"] = int(should_do_ttt)
                log_dict["gate/ttt_remaining"] = ph_gate.ttt_remaining
                log_dict["gate/total_triggers"] = ph_gate.total_triggers
                log_dict["gate/ema_entropy"] = ph_gate.ema if ph_gate.ema else 0.0
                log_dict["gate/ttt_skipped"] = int(result.get('ttt_skipped', False))
            wandb.log(log_dict, step=n)
        
        # Detect domain boundary and print per-domain summary
        next_domain = questions[i + 1]['domain'] if i + 1 < len(questions) else None
        if next_domain != domain:
            # Finished a domain - print detailed per-domain metrics
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
                
                # Log domain completion to W&B
                if use_wandb:
                    wandb.log({
                        f"domain_final/{domain}/acc": d_acc,
                        f"domain_final/{domain}/brier": d_brier,
                        f"domain_final/{domain}/ece": d_ece,
                        f"domain_final/{domain}/auroc": d_auroc,
                    }, step=n, commit=False)
        
        # Intermediate summary every 10 questions
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
    
    # Compute per-domain metrics (AUROC, Brier, ECE)
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
    
    # PH Gate summary
    if args.use_ph_gate and ph_gate is not None:
        ttt_trained = sum(1 for r in results if not r.get('ttt_skipped', False))
        ttt_skipped = sum(1 for r in results if r.get('ttt_skipped', False))
        print(f"\nPH Gate Summary:")
        print(f"  Total triggers: {ph_gate.total_triggers}")
        print(f"  TTT rounds: {ttt_trained} ({100*ttt_trained/len(results):.1f}%)")
        print(f"  Skipped: {ttt_skipped} ({100*ttt_skipped/len(results):.1f}%)")
    
    # P(True) distribution
    all_p_trues = [p for r in results for p in r.get('neighbor_p_trues', [])]
    if all_p_trues:
        print(f"\nP(True) Distribution:")
        print(f"  Mean: {np.mean(all_p_trues):.3f}")
        print(f"  Median: {np.median(all_p_trues):.3f}")
        print(f"  Std: {np.std(all_p_trues):.3f}")
        print(f"  Range: [{min(all_p_trues):.3f}, {max(all_p_trues):.3f}]")
    
    # Final W&B logging
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
        
        # Log gate final stats
        if args.use_ph_gate and ph_gate is not None:
            ttt_trained = sum(1 for r in results if not r.get('ttt_skipped', False))
            log_dict["final/gate_triggers"] = ph_gate.total_triggers
            log_dict["final/ttt_trained_pct"] = ttt_trained / len(results)
        
        wandb.log(log_dict, step=len(questions))
        try:
            wandb.finish(quiet=True)
        except Exception:
            pass
    
    # Save results
    output_file = f"continual_ttt_{args.mode}_results.json"
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
