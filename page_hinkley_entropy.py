#!/usr/bin/env python3
"""Run entropy-only inference over a mixed-domain stream and plot Page-Hinkley detection."""

from __future__ import annotations

import argparse
from typing import List, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from matplotlib import pyplot as plt
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_texts(domain: str, n: int) -> List[str]:
    def _safe_select(ds, n_req: int, name: str):
        n_avail = len(ds)
        n_take = min(n_req, n_avail)
        if n_take < n_req:
            print(f"[warn] {name}: requested n={n_req}, but dataset has n={n_avail}; using n={n_take}")
        return ds.select(range(n_take)), n_take

    if domain == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main", split="train")
        ds, _ = _safe_select(ds, n, "gsm8k")
        return [ex["question"] for ex in ds]

    if domain == "mmlu":
        subjects = [
            "abstract_algebra", "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology", "college_chemistry",
            "college_computer_science", "college_mathematics", "college_physics",
            "computer_security", "conceptual_physics", "econometrics",
            "electrical_engineering", "elementary_mathematics", "formal_logic",
            "global_facts", "high_school_biology", "high_school_chemistry",
            "high_school_computer_science", "high_school_european_history",
            "high_school_geography", "high_school_government_and_politics",
            "high_school_macroeconomics", "high_school_mathematics",
            "high_school_microeconomics", "high_school_physics",
            "high_school_psychology", "high_school_statistics",
            "high_school_us_history", "high_school_world_history",
            "human_aging", "human_sexuality", "international_law",
            "jurisprudence", "logical_fallacies", "machine_learning",
            "management", "marketing", "medical_genetics", "miscellaneous",
            "moral_disputes", "moral_scenarios", "nutrition", "philosophy",
            "prehistory", "professional_accounting", "professional_law",
            "professional_medicine", "professional_psychology", "public_relations",
            "security_studies", "sociology", "us_foreign_policy", "virology",
            "world_religions",
        ]
        iters: Dict[str, any] = {}
        exhausted = set()
        texts: List[str] = []
        subj_idx = 0
        while len(texts) < n and len(exhausted) < len(subjects):
            subj = subjects[subj_idx % len(subjects)]
            subj_idx += 1
            if subj not in iters and subj not in exhausted:
                try:
                    iters[subj] = iter(load_dataset("cais/mmlu", subj, split="test"))
                except Exception as e:
                    print(f"[warn] mmlu/{subj}: {e}")
                    exhausted.add(subj)
                    continue
            if subj in exhausted:
                continue
            try:
                ex = next(iters[subj])
            except StopIteration:
                exhausted.add(subj)
                del iters[subj]
                continue
            q = ex.get("question", "")
            if q:
                texts.append(q)
        if len(texts) < n:
            print(f"[warn] mmlu: collected {len(texts)} / {n}")
        return texts

    if domain == "arc":
        ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        ds, n_take = _safe_select(ds, n, "arc/ARC-Challenge")
        texts: List[str] = []
        for ex in ds:
            choices = "\n".join(f"{l}. {t}" for l, t in zip(ex["choices"]["label"], ex["choices"]["text"]))
            texts.append(ex["question"] + "\n" + choices)
        return texts[:n_take]

    if domain == "truthfulqa_mc":
        ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
        ds, n_take = _safe_select(ds, n, "truthfulqa_mc")
        texts: List[str] = []
        for ex in ds:
            choices = "\n".join(f"{i+1}. {c}" for i, c in enumerate(ex["mc1_targets"]["choices"]))
            texts.append(ex["question"] + "\n" + choices)
        return texts[:n_take]

    raise ValueError(f"Unknown domain: {domain}")


def build_stream(domains: List[str], samples_per_domain: int) -> Tuple[List[str], List[str], Dict[str, int]]:
    stream_texts: List[str] = []
    stream_labels: List[str] = []
    block_sizes: Dict[str, int] = {}
    for d in tqdm(domains, desc="Loading domains"):
        texts = load_texts(d, samples_per_domain)
        stream_texts.extend(texts)
        stream_labels.extend([d] * len(texts))
        block_sizes[d] = len(texts)
    return stream_texts, stream_labels, block_sizes


def entropy_last_token(model, enc) -> float:
    with torch.no_grad():
        out = model(**enc)
        attn = enc["attention_mask"]
        last_idx = attn.sum(dim=1) - 1
        b_idx = torch.arange(attn.size(0), device=attn.device)
        logits = out.logits[b_idx, last_idx, :]
        logp = F.log_softmax(logits, dim=-1)
        ent = -(logp.exp() * logp).sum(dim=-1)
        return float(ent.mean().item())


def ema_smooth(data: List[float], alpha: float) -> List[float]:
    """Apply exponential moving average smoothing."""
    smoothed = []
    ema = data[0]
    for x in data:
        ema = alpha * x + (1 - alpha) * ema
        smoothed.append(ema)
    return smoothed


def page_hinkley_stats(data: List[float], delta: float, threshold: float) -> Tuple[List[float], List[int], List[float]]:
    """Bidirectional Page-Hinkley test detecting both upward and downward shifts.
    
    Returns:
        ph_stat: max(ph_up, ph_down) at each step (for plotting)
        change_points: indices where a change was detected
        running_mean: segment running mean at each step
    """
    ph_stat: List[float] = []
    change_points: List[int] = []
    running_mean: List[float] = []
    
    segment_start = 0
    m_up = 0.0
    m_up_min = 0.0
    m_down = 0.0
    m_down_min = 0.0
    
    for t, value in enumerate(data):
        segment_data = data[segment_start:t+1]
        mean_t = np.mean(segment_data)
        running_mean.append(mean_t)
        
        # Upward shift: detect when value >> mean
        dev_up = value - mean_t - delta
        m_up += dev_up
        if t == segment_start:
            m_up_min = m_up
        else:
            m_up_min = min(m_up_min, m_up)
        ph_u = m_up - m_up_min
        
        # Downward shift: detect when value << mean
        dev_down = mean_t - value - delta
        m_down += dev_down
        if t == segment_start:
            m_down_min = m_down
        else:
            m_down_min = min(m_down_min, m_down)
        ph_d = m_down - m_down_min
        
        ph_stat.append(max(ph_u, ph_d))
        
        # Trigger on either direction
        if ph_u > threshold or ph_d > threshold:
            change_points.append(t)
            segment_start = t + 1
            m_up = m_down = 0.0
            m_up_min = m_down_min = 0.0
    
    return ph_stat, change_points, running_mean


def main() -> None:
    parser = argparse.ArgumentParser(description="Page-Hinkley entropy stream plot.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct")
    parser.add_argument("--domains", nargs="+", default=["gsm8k", "mmlu", "arc", "truthfulqa_mc"])
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--ph_delta", type=float, default=0.05)
    parser.add_argument("--ph_threshold", type=float, default=4.0)
    parser.add_argument("--ema_alpha", type=float, default=0.05, help="EMA smoothing factor (lower=smoother)")
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
    ).to(device)
    model.eval()

    stream_texts, stream_labels, block_sizes = build_stream(args.domains, args.samples)
    
    # Collect entropies with progress bar
    entropies: List[float] = []
    for text in tqdm(stream_texts, desc="Computing entropies", unit="sample"):
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_length,
            padding=False,
        ).to(device)
        entropies.append(entropy_last_token(model, enc))

    # Apply EMA smoothing
    smoothed_entropies = ema_smooth(entropies, args.ema_alpha)
    print(f"Applied EMA smoothing with alpha={args.ema_alpha}")

    # Run Page-Hinkley on smoothed signal
    ph_stat, ph_triggers, _ = page_hinkley_stats(smoothed_entropies, args.ph_delta, args.ph_threshold)
    print(f"PH change points ({len(ph_triggers)}): {ph_triggers}")

    if args.plot:
        xs = list(range(len(entropies)))
        fig, (ax_ent, ax_ph) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        
        # Helper to draw domain boundaries
        def draw_domain_lines(ax):
            cumsum = 0
            for d in args.domains[:-1]:
                cumsum += block_sizes.get(d, 0)
                ax.axvline(cumsum, color="blue", linestyle="--", alpha=0.6, label="domain boundary" if cumsum == block_sizes.get(args.domains[0], 0) else "")
        
        # Top: entropy (raw + smoothed)
        ax_ent.plot(xs, entropies, color="lightgray", linewidth=0.5, alpha=0.7, label="raw entropy")
        ax_ent.plot(xs, smoothed_entropies, color="black", linewidth=1.2, label=f"EMA (α={args.ema_alpha})")
        if ph_triggers:
            ax_ent.scatter(ph_triggers, [smoothed_entropies[i] for i in ph_triggers], color="red", marker="x", s=80, zorder=5, label="PH trigger")
        draw_domain_lines(ax_ent)
        ax_ent.set_ylabel("entropy")
        ax_ent.set_title("Entropy stream (raw + EMA smoothed)")
        ax_ent.legend()
        ax_ent.grid(True, alpha=0.3)
        
        # Bottom: PH statistic
        ax_ph.plot(xs, ph_stat, color="purple", linewidth=0.8, label="PH statistic")
        ax_ph.axhline(args.ph_threshold, color="orange", linestyle="--", alpha=0.8, label=f"threshold={args.ph_threshold}")
        if ph_triggers:
            ax_ph.scatter(ph_triggers, [ph_stat[i] for i in ph_triggers], color="red", marker="x", s=80, label="PH trigger")
        draw_domain_lines(ax_ph)
        ax_ph.set_xlabel("sample index")
        ax_ph.set_ylabel("PH statistic")
        ax_ph.set_title("Page-Hinkley statistic (bidirectional)")
        ax_ph.legend()
        ax_ph.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig("page_hinkley_entropy.png", dpi=150, bbox_inches="tight")
        print("Saved page_hinkley_entropy.png")


if __name__ == "__main__":
    main()
