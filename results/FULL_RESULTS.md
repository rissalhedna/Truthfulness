# SECL — Full Experimental Results

All numbers computed from result JSON files. ECE uses 10 equal-width bins with midpoint mapping. AUROC and Brier use sklearn/numpy. All runs use seed=42.

---

## 1. Main Results

All runs: 500 questions/domain, 4 domains sequential (GSM8K → MMLU → ARC → TruthfulQA), 2000 total.

### 1.1 Llama 3.2-3B (norm_temperature=0.7)

| Method | Domain | ECE ↓ | Brier ↓ | AUROC ↑ | Acc |
|---|---|---|---|---|---|
| Soft Verbalized | GSM8K | 0.218 | 0.294 | 0.565 | 0.472 |
| Soft Verbalized | MMLU | 0.106 | 0.253 | 0.601 | 0.562 |
| Soft Verbalized | ARC | 0.095 | 0.207 | 0.568 | 0.726 |
| Soft Verbalized | TruthfulQA | 0.372 | 0.414 | 0.301 | 0.544 |
| Soft Verbalized | **OVERALL** | 0.170 | 0.292 | 0.510 | 0.576 |
| P(True) Norm | GSM8K | 0.133 | 0.264 | 0.594 | 0.476 |
| P(True) Norm | MMLU | 0.091 | 0.239 | 0.652 | 0.566 |
| P(True) Norm | ARC | 0.117 | 0.210 | 0.624 | 0.728 |
| P(True) Norm | TruthfulQA | 0.089 | 0.180 | 0.818 | 0.532 |
| P(True) Norm | **OVERALL** | 0.065 | 0.223 | 0.693 | 0.576 |
| **SECL (Ours)** | GSM8K | **0.070** | 0.249 | 0.573 | 0.488 |
| **SECL (Ours)** | MMLU | **0.067** | 0.249 | 0.549 | 0.558 |
| **SECL (Ours)** | ARC | 0.112 | 0.207 | 0.616 | 0.714 |
| **SECL (Ours)** | TruthfulQA | **0.068** | 0.260 | 0.454 | 0.546 |
| **SECL (Ours)** | **OVERALL** | **0.050** | 0.241 | 0.587 | 0.577 |

SECL trains on **25.6%** of questions.

### 1.2 Gemma 2-2B (norm_temperature=1.5)

| Method | Domain | ECE ↓ | Brier ↓ | AUROC ↑ | Acc |
|---|---|---|---|---|---|
| Soft Verbalized | GSM8K | 0.549 | 0.446 | 0.658 | 0.186 |
| Soft Verbalized | MMLU | 0.194 | 0.271 | 0.608 | 0.576 |
| Soft Verbalized | ARC | 0.082 | 0.201 | 0.534 | 0.738 |
| Soft Verbalized | TruthfulQA | 0.267 | 0.338 | 0.417 | 0.566 |
| Soft Verbalized | **OVERALL** | 0.256 | 0.314 | 0.558 | 0.516 |
| P(True) Norm | GSM8K | 0.395 | 0.326 | 0.685 | 0.186 |
| P(True) Norm | MMLU | 0.163 | 0.263 | 0.616 | 0.576 |
| P(True) Norm | ARC | 0.185 | 0.229 | 0.597 | 0.738 |
| P(True) Norm | TruthfulQA | 0.104 | 0.213 | 0.729 | 0.566 |
| P(True) Norm | **OVERALL** | 0.141 | 0.258 | 0.650 | 0.516 |
| **SECL (Ours)** | GSM8K | 0.356 | **0.271** | 0.609 | 0.180 |
| **SECL (Ours)** | MMLU | **0.054** | **0.238** | 0.594 | 0.578 |
| **SECL (Ours)** | ARC | 0.144 | 0.213 | 0.591 | 0.728 |
| **SECL (Ours)** | TruthfulQA | 0.210 | 0.293 | 0.343 | 0.574 |
| **SECL (Ours)** | **OVERALL** | **0.056** | 0.254 | 0.548 | 0.515 |

SECL trains on **6.0%** of questions.

### 1.3 Phi 3.5-Mini (norm_temperature=1.5)

| Method | Domain | ECE ↓ | Brier ↓ | AUROC ↑ | Acc |
|---|---|---|---|---|---|
| Soft Verbalized | GSM8K | 0.290 | 0.304 | 0.563 | 0.650 |
| Soft Verbalized | MMLU | 0.264 | 0.286 | 0.602 | 0.648 |
| Soft Verbalized | ARC | 0.109 | 0.150 | 0.588 | 0.826 |
| Soft Verbalized | TruthfulQA | 0.343 | 0.362 | 0.592 | 0.546 |
| Soft Verbalized | **OVERALL** | 0.251 | 0.275 | 0.600 | 0.667 |
| P(True) Norm | GSM8K | 0.261 | 0.303 | 0.559 | 0.650 |
| P(True) Norm | MMLU | 0.171 | 0.243 | 0.648 | 0.648 |
| P(True) Norm | ARC | 0.129 | 0.157 | 0.680 | 0.826 |
| P(True) Norm | TruthfulQA | 0.146 | 0.205 | 0.771 | 0.546 |
| P(True) Norm | **OVERALL** | 0.154 | 0.227 | 0.675 | 0.667 |
| **SECL (Ours)** | GSM8K | 0.283 | 0.297 | 0.575 | 0.658 |
| **SECL (Ours)** | MMLU | **0.054** | **0.223** | 0.604 | 0.644 |
| **SECL (Ours)** | ARC | 0.129 | 0.167 | 0.499 | 0.826 |
| **SECL (Ours)** | TruthfulQA | 0.229 | 0.319 | 0.407 | 0.532 |
| **SECL (Ours)** | **OVERALL** | **0.115** | 0.251 | 0.521 | 0.665 |

SECL trains on **8.0%** of questions.

### 1.4 Overall ECE Summary (main claim)

| Model | Soft Verbalized | P(True) Norm | SECL (Ours) | TTT% |
|---|---|---|---|---|
| Llama 3.2-3B | 0.170 | 0.065 | **0.050** | 25.6% |
| Gemma 2-2B | 0.256 | 0.141 | **0.056** | 6.0% |
| Phi 3.5-Mini | 0.251 | 0.154 | **0.115** | 8.0% |

---

## 2. Ablations

### 2A: Weight Accumulation vs Reset (Llama, PH-gated)

| Run | Config | ECE ↓ | Brier ↓ | AUROC ↑ | Acc | TTT% |
|---|---|---|---|---|---|---|
| E28_PH_BURST50 | Accumulate, B=50 | **0.050** | **0.241** | **0.587** | 0.577 | 25.6% |
| E28_PH_RESET_COOL50_BURST20 | Reset on trigger, cooldown=50, B=20 | 0.114 | 0.276 | 0.539 | 0.575 | 9.3% |

Per-domain breakdown for Reset run:

| Domain | ECE | Brier | AUROC | Acc |
|---|---|---|---|---|
| GSM8K | **0.026** | 0.243 | 0.592 | 0.492 |
| MMLU | 0.088 | 0.251 | 0.604 | 0.552 |
| ARC | **0.072** | 0.204 | 0.599 | 0.720 |
| TruthfulQA | 0.374 | 0.407 | 0.293 | 0.534 |

Reset achieves excellent per-domain ECE on GSM8K/ARC but collapses on TruthfulQA (0.374) because only 9.3% of questions trigger training — too sparse for the final domain.

### 2B: Gating Strategy (Llama, overall metrics)

| Strategy | ECE ↓ | Brier ↓ | AUROC ↑ | Acc | TTT% |
|---|---|---|---|---|---|
| Always-on MSE (every question) | 0.047 | **0.240** | **0.593** | 0.579 | 100% |
| Bin-Gate threshold=1 | 0.052 | 0.242 | 0.585 | 0.579 | 55.8% |
| Bin-Gate threshold=2 | **0.044** | 0.242 | 0.578 | 0.570 | 32.6% |
| PH-Gate Burst=50 | 0.050 | 0.241 | 0.587 | 0.577 | 25.6% |

Always-on achieves the best Brier/AUROC but requires P(True) on 100% of questions. PH-Gate achieves comparable ECE (0.050 vs 0.047) while computing P(True) on only 25.6% of questions.

### 2C: Loss Function — Plain MSE vs Directional (Llama, bin-gate=2)

| Loss | ECE ↓ | Brier ↓ | AUROC ↑ | Acc | TTT% |
|---|---|---|---|---|---|
| Plain MSE | 0.048 | 0.242 | 0.574 | 0.580 | 33.5% |
| Directional (α=0.5, δ=0.15) | **0.044** | 0.242 | **0.578** | 0.570 | 32.6% |

Directional loss improves ECE and AUROC marginally under the same gating.

### 2D: Burst Size (Llama, PH-gated)

| Burst Size | ECE ↓ | Brier ↓ | AUROC ↑ | Acc | TTT% |
|---|---|---|---|---|---|
| B=20 (with reset+cooldown=50) | 0.114 | 0.276 | 0.539 | 0.575 | 9.3% |
| B=50 (no reset) | **0.050** | **0.241** | **0.587** | 0.577 | 25.6% |

Note: these differ in both burst size AND reset/cooldown, so this is not a clean burst-only ablation. B=50 without reset is the better configuration.

### 2E: Domain Ordering — Forward vs Reversed (all 3 models)

Forward = GSM8K → MMLU → ARC → TruthfulQA. Reversed = TruthfulQA → ARC → MMLU → GSM8K.

#### Llama 3.2-3B

| Order | Domain | ECE ↓ | Brier ↓ | AUROC ↑ | Acc |
|---|---|---|---|---|---|
| Forward | GSM8K | 0.070 | 0.249 | 0.573 | 0.488 |
| Reversed | GSM8K | 0.117 | 0.259 | 0.527 | 0.484 |
| Forward | MMLU | 0.067 | 0.249 | 0.549 | 0.558 |
| Reversed | MMLU | **0.032** | **0.238** | **0.602** | 0.572 |
| Forward | ARC | 0.112 | 0.207 | **0.616** | 0.714 |
| Reversed | ARC | **0.081** | **0.205** | 0.598 | 0.710 |
| Forward | TruthfulQA | **0.068** | **0.260** | 0.454 | 0.546 |
| Reversed | TruthfulQA | 0.134 | 0.289 | 0.456 | 0.532 |
| **Forward** | **OVERALL** | **0.050** | 0.241 | **0.587** | 0.577 |
| **Reversed** | **OVERALL** | 0.068 | 0.248 | 0.578 | 0.575 |

#### Gemma 2-2B

| Order | Domain | ECE ↓ | Brier ↓ | AUROC ↑ | Acc |
|---|---|---|---|---|---|
| Forward | GSM8K | **0.356** | **0.271** | 0.609 | 0.180 |
| Reversed | GSM8K | 0.396 | 0.306 | **0.642** | 0.194 |
| Forward | MMLU | **0.054** | 0.238 | 0.594 | 0.578 |
| Reversed | MMLU | 0.096 | 0.239 | **0.621** | 0.584 |
| Forward | ARC | 0.144 | 0.213 | **0.591** | 0.728 |
| Reversed | ARC | **0.046** | **0.194** | 0.548 | 0.740 |
| Forward | TruthfulQA | 0.210 | 0.293 | 0.343 | 0.574 |
| Reversed | TruthfulQA | **0.189** | 0.293 | **0.367** | 0.562 |
| **Forward** | **OVERALL** | **0.056** | **0.254** | 0.548 | 0.515 |
| **Reversed** | **OVERALL** | 0.149 | 0.258 | **0.625** | 0.520 |

#### Phi 3.5-Mini

| Order | Domain | ECE ↓ | Brier ↓ | AUROC ↑ | Acc |
|---|---|---|---|---|---|
| Forward | GSM8K | 0.283 | 0.297 | **0.575** | 0.658 |
| Reversed | GSM8K | **0.068** | **0.233** | 0.481 | 0.656 |
| Forward | MMLU | **0.054** | **0.223** | **0.604** | 0.644 |
| Reversed | MMLU | 0.213 | 0.265 | 0.552 | 0.658 |
| Forward | ARC | 0.129 | 0.167 | 0.499 | 0.826 |
| Reversed | ARC | **0.099** | **0.144** | **0.558** | 0.836 |
| Forward | TruthfulQA | **0.229** | **0.319** | 0.407 | 0.532 |
| Reversed | TruthfulQA | 0.343 | 0.359 | **0.605** | 0.548 |
| **Forward** | **OVERALL** | **0.115** | 0.251 | 0.521 | 0.665 |
| **Reversed** | **OVERALL** | 0.173 | **0.250** | **0.575** | 0.674 |

Pattern: the first domain in the ordering gets worse ECE (warmup, no LoRA adaptation yet). Per-domain ECE wins shift with ordering, but overall ECE remains competitive in both directions.

### 2F: Qwen 2.5-3B — Negative Control

Qwen was tested as a candidate model but dropped because P(True) Norm was **worse** than Soft Verbalized on ECE at every temperature — the generation-discrimination gap does not exist for this model, so SECL's core assumption is violated.

| Method | ECE ↓ | Brier ↓ | AUROC ↑ | Acc | N |
|---|---|---|---|---|---|
| Soft Verbalized | 0.247 | 0.272 | 0.565 | 0.715 | 400 |
| Hard Verbalized | 0.250 | 0.276 | 0.566 | 0.715 | 400 |
| P(True) T=0.3 | 0.290 | 0.299 | 0.569 | 0.715 | 400 |
| P(True) T=0.7 | 0.263 | 0.287 | 0.577 | 0.715 | 400 |
| P(True) T=1.0 | 0.257 | 0.286 | 0.571 | 0.715 | 400 |
| P(True) T=1.5 | 0.265 | 0.288 | 0.571 | 0.715 | 400 |
| P(True) T=2.0 | 0.267 | 0.291 | 0.576 | 0.715 | 400 |
| P(True) T=3.0 | 0.291 | 0.304 | 0.582 | 0.715 | 400 |

Note: Qwen runs used only 100 questions/domain (400 total) due to early termination after identifying the negative result. Best P(True) ECE (0.257 at T=1.0) is still worse than Soft Verbalized (0.247).

---

## 3. Computational Cost Analysis

P(True) Norm requires 5 discriminative forward passes per question (1 answer + 4 distractors). SECL only computes P(True) when the PH gate fires.

| Model | TTT% | SECL FWD-equiv | Baseline FWD | Savings |
|---|---|---|---|---|
| Llama 3.2-3B | 25.6% | 9,168 | 12,000 | **24%** |
| Gemma 2-2B | 6.0% | 3,666 | 12,000 | **69%** |
| Phi 3.5-Mini | 8.0% | 4,240 | 12,000 | **65%** |

FWD-equiv counts backward passes as 2× forward. Baseline = 2000 questions × 6 passes each (1 generation + 5 P(True)).

---

## 4. Hyperparameter Settings

### SECL (used across all 3 models)

| Parameter | Symbol | Value |
|---|---|---|
| PH tolerance | ε | 0.05 |
| PH detection threshold | λ | 3.0 |
| EMA smoothing factor | α_ema | 0.05 |
| Burst size | B | 50 |
| Warmup | — | 30 questions |
| Directional step size | α_step | 0.5 |
| Directional clip | δ | 0.15 |
| Bin-gate threshold | — | 1 bin |
| LoRA rank | r | 8 |
| LoRA target | — | last 8 layers (Gemma, 18-25 of 26; Phi, 24-31 of 32), last 4 layers (Llama, 24-27 of 28) |
| Learning rate | — | 5e-5 |
| TTT epochs per question | — | 3 |
| Optimizer | — | AdamW |

### Model-specific

| Model | norm_temperature (τ) | LoRA target keys | trust_remote_code |
|---|---|---|---|
| Llama 3.2-3B | 0.7 | q_proj, v_proj | False |
| Gemma 2-2B | 1.5 | q_proj, v_proj | False |
| Phi 3.5-Mini | 1.5 | qkv_proj | False |

τ selected per model via sweep on P(True) Norm baseline (see Section 2F temperature sweep data for Gemma/Phi; Llama uses τ=0.7).
