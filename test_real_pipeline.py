#!/usr/bin/env python3
"""
REAL end-to-end pipeline test using ACTUAL code from run_continual_ttt.py
with a real model. Tests every stage and logs every intermediate value.

No mocks. No fakes. The actual pipeline.
"""
import os, sys, re
sys.path.insert(0, os.path.dirname(__file__))

# Minimal args to bootstrap the module — we parse before importing
# run_continual_ttt.py because it does argparse at module level
sys.argv = [
    'test_real_pipeline.py',
    '--gpu', '3',
    '--mode', 'sequential',
    '--model_name', 'meta-llama/Llama-3.2-3B-Instruct',
    '--neighbor_model', 'meta-llama/Llama-3.2-3B-Instruct',
    '--questions_per_domain', '2',
    '--n_neighbors', '2',
    '--n_epochs', '1',
    '--k_distractors', '4',
    '--reuse_mcq_options',
    '--norm_temperature', '0.7',
    '--no_wandb',
    '--seed', '42',
    '--num_bins', '10',
    '--bin_window', '500',
    '--bin_min_count', '5',
    '--layer_start', '14',
    '--layer_end', '16',
    '--lr', '5e-5',
]

# ── Now import everything from the actual codebase ──
import run_continual_ttt as R
import torch
import numpy as np

# Shortcuts to actual functions
_is_mc_answer = R._is_mc_answer
generate_neighborhood_questions = R.generate_neighborhood_questions
generate_distractors = R.generate_distractors
get_discriminative_confidence = R.get_discriminative_confidence
normalize_p_true_sharpened = R.normalize_p_true_sharpened
get_baseline_answer = R.get_baseline_answer
create_qa_prompt = R.create_qa_prompt
extract_answer = R.extract_answer
extract_confidence = R.extract_confidence
create_confidence_mask = R.create_confidence_mask
get_bin_digit_tokens = R.get_bin_digit_tokens
train_single_question_discriminative = R.train_single_question_discriminative
answers_match = R.answers_match
bin_mapper = R.bin_mapper
CONF_TO_PROB = R.CONF_TO_PROB
args = R.args

model = R.base_model
tokenizer = R.tokenizer

from peft import LoraConfig, get_peft_model, TaskType

# ── Load a few real MMLU questions (MCQ with mc_options) ──
from utils import load_qa_dataset
print("\n" + "=" * 70)
print("  LOADING 3 REAL MMLU QUESTIONS")
print("=" * 70)
mmlu_qs = load_qa_dataset('mmlu', num_samples=3)
for i, q in enumerate(mmlu_qs):
    print(f"  Q{i}: {q['question'][:80]}...")
    print(f"       GT: {q['ground_truth_answer']}  mc_options: {list(q.get('mc_options', {}).keys())}")
print()

# Also load 1 GSM8K question (no mc_options)
gsm_qs = load_qa_dataset('gsm8k', num_samples=1)
print(f"  GSM8K Q0: {gsm_qs[0]['question'][:80]}...")
print(f"       GT: {gsm_qs[0]['ground_truth_answer']}  mc_options: {gsm_qs[0].get('mc_options')}")
print()

# ═══════════════════════════════════════════════════════════════════
# TEST 1: BASELINE PATH — MCQ question
# ═══════════════════════════════════════════════════════════════════
qa = mmlu_qs[0]
question = qa['question']
mc_opts = qa.get('mc_options')

print("=" * 70)
print("  TEST 1: BASELINE PATH (MCQ, MMLU)")
print("=" * 70)
print(f"  Question: {question[:120]}...")
print(f"  mc_options: {mc_opts}")
print()

# Step 1: Get baseline answer
print("--- Step 1: Baseline answer ---")
baseline = get_baseline_answer(question, model, tokenizer)
print(f"  Answer: {baseline['answer']!r}")
print(f"  Confidence: {baseline['confidence']}")
print(f"  Response: {baseline['response'][:200]}")
print()

# Step 2: Check if answer is MCQ letter
print("--- Step 2: MCQ letter check ---")
ans_letter = _is_mc_answer(baseline['answer'])
print(f"  _is_mc_answer({baseline['answer']!r}) = {ans_letter}")
print()

# Step 3: Build candidates (baseline_use_ptrue_norm path)
print("--- Step 3: Build candidates ---")
if mc_opts and ans_letter and ans_letter in mc_opts:
    candidates = [f"{ans_letter}. {mc_opts[ans_letter]}"]
    for ltr in sorted(mc_opts.keys()):
        if ltr != ans_letter:
            candidates.append(f"{ltr}. {mc_opts[ltr]}")
    method = "MCQ structured"
else:
    distractors = generate_distractors(question, baseline['answer'], model, tokenizer, k=args.k_distractors, mc_options=mc_opts)
    candidates = [baseline['answer']] + distractors
    method = "FALLBACK (generated distractors)"
print(f"  Method: {method}")
print(f"  Candidates ({len(candidates)}):")
for j, c in enumerate(candidates):
    print(f"    [{j}] {c}")
print()

# Step 4: P(True) for each candidate
print("--- Step 4: P(True) per candidate ---")
with torch.no_grad():
    p_scores = []
    for cand in candidates:
        p = get_discriminative_confidence(question, cand, baseline['response'], model, tokenizer,
                                          use_p_know=False, claimed_bin=None, candidate_list=candidates)
        p_scores.append(p)
        print(f"    P(True) for {cand[:40]:40s} = {p:.4f}")
print()

# Step 5: Normalize
print("--- Step 5: Normalize P(True) ---")
p_true_norm = normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
print(f"  p_true_raw (answer): {p_scores[0]:.4f}")
print(f"  p_true_norm (T={args.norm_temperature}): {p_true_norm:.4f}")
print(f"  bin: {bin_mapper.to_bin(p_true_norm)}")
print()

# Check for duplicates
print("--- Step 6: Duplicate check ---")
if len(candidates) != len(set(candidates)):
    print(f"  !! DUPLICATE CANDIDATES FOUND: {candidates}")
else:
    print(f"  OK — no duplicates in {len(candidates)} candidates")
print()


# ═══════════════════════════════════════════════════════════════════
# TEST 2: FULL TTT PATH — with neighbors + reuse_mcq_options
# ═══════════════════════════════════════════════════════════════════
qa = mmlu_qs[1]
question = qa['question']
mc_opts = qa.get('mc_options')

print("=" * 70)
print("  TEST 2: FULL TTT PATH (MCQ + 2 neighbors + reuse_mcq)")
print("=" * 70)
print(f"  Question: {question[:120]}...")
print(f"  mc_options: {mc_opts}")
print(f"  reuse_mcq_options: {args.reuse_mcq_options}")
print()

# Build LoRA model for TTT
print("--- Setting up LoRA ---")
target_modules = R.get_late_layers_target_modules(args.layer_start, args.layer_end)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    lora_dropout=0.05,
    bias="none",
)
train_model = get_peft_model(model, lora_config)
optimizer = torch.optim.AdamW(train_model.parameters(), lr=args.lr)
train_model.eval()
print(f"  LoRA modules: {len(target_modules)} targets on layers {args.layer_start}-{args.layer_end-1}")
print()

# Step 1: Generate neighbors
print("--- Step 1: Generate neighbors (base model, no adapter) ---")
with train_model.disable_adapter():
    neighbors = generate_neighborhood_questions(question, train_model, tokenizer, n_neighbors=2)
for j, n in enumerate(neighbors):
    print(f"  Neighbor {j}: {n['question'][:80]}...")
print()

# Step 2: Build questions_to_train
questions_to_train = [{'question': question, 'difficulty': 'input'}] + neighbors
print(f"--- Step 2: questions_to_train ({len(questions_to_train)} items) ---")
for j, qt in enumerate(questions_to_train):
    print(f"  [{j}] ({qt['difficulty']}) {qt['question'][:60]}...")
print()

# Step 3: Pass 1 — signal collection (manually, to log everything)
print("--- Step 3: Pass 1 — Signal collection ---")
neighbor_data = []
for j, n in enumerate(questions_to_train):
    q = n['question']
    difficulty = n['difficulty']
    
    # THE FIX: append MCQ options to neighbor questions
    if mc_opts and args.reuse_mcq_options and difficulty != 'input':
        choices = "\n".join(f"{ltr}. {mc_opts[ltr]}" for ltr in sorted(mc_opts))
        q = f"{q}\n{choices}"
        print(f"  [{j}] ({difficulty}) Appended MCQ options to prompt")
    else:
        print(f"  [{j}] ({difficulty}) Original question (already has options or is input)")
    
    # Generate answer
    prompt = create_qa_prompt(q)
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
    with torch.no_grad():
        outputs = train_model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    answer = extract_answer(response)
    claimed_conf = extract_confidence(response)
    del outputs
    
    print(f"    Response: {response[:300]}")
    print(f"    Answer: {answer!r}")
    print(f"    Confidence: {claimed_conf}")
    
    # P(Know)
    with torch.no_grad():
        p_know = get_discriminative_confidence(q, answer, response, train_model, tokenizer,
                                                use_p_know=True, claimed_bin=claimed_conf)
    print(f"    P(Know): {p_know:.4f}")
    
    # MCQ letter check
    use_mc = mc_opts if (mc_opts and (args.reuse_mcq_options or difficulty == 'input')) else None
    ans_letter = _is_mc_answer(answer) if use_mc else None
    print(f"    _is_mc_answer: {ans_letter}  |  use_mc: {'yes' if use_mc else 'no'}")
    
    # Build candidates
    if use_mc and ans_letter and ans_letter in use_mc:
        candidates = [f"{ans_letter}. {use_mc[ans_letter]}"]
        for ltr in sorted(use_mc.keys()):
            if ltr != ans_letter:
                candidates.append(f"{ltr}. {use_mc[ltr]}")
        cand_method = "MCQ structured"
    else:
        distractors = generate_distractors(q, answer, train_model, tokenizer, k=args.k_distractors, mc_options=mc_opts)
        candidates = [answer] + distractors
        cand_method = "FALLBACK (generated distractors)"
    
    print(f"    Candidate method: {cand_method}")
    print(f"    Candidates: {candidates}")
    
    # Check duplicates
    if len(candidates) != len(set(candidates)):
        print(f"    !! DUPLICATES FOUND !!")
    
    # P(True) scores
    p_scores = []
    with torch.no_grad():
        for cand in candidates:
            p = get_discriminative_confidence(q, cand, response, train_model, tokenizer,
                                              use_p_know=False, claimed_bin=None, candidate_list=candidates)
            p_scores.append(p)
    
    p_true_raw = p_scores[0]
    no_distractors = len(candidates) <= 1
    if no_distractors:
        p_true_norm = p_true_raw
    else:
        p_true_norm = normalize_p_true_sharpened(p_scores, temperature=args.norm_temperature)
    
    print(f"    P(True) scores: {[f'{s:.3f}' for s in p_scores]}")
    print(f"    p_true_norm (T={args.norm_temperature}): {p_true_norm:.4f}")
    
    neighbor_data.append({
        'question': q, 'difficulty': difficulty, 'answer': answer,
        'response': response, 'p_true_raw': p_true_raw, 'p_true_norm': p_true_norm,
        'p_know': p_know, 'no_distractors': no_distractors,
    })
    print()

# Pass 2: Derived values
print("--- Pass 2: Derived values ---")
pknow_values = [nd['p_know'] for nd in neighbor_data]
p_know_mean = float(np.mean(pknow_values))
print(f"  p_know_mean (global): {p_know_mean:.4f}")
for nd in neighbor_data:
    nd['p_fused'] = nd['p_true_norm'] * p_know_mean
    p_for_bin = nd['p_true_norm']  # default mode
    print(f"  [{nd['difficulty']}] p_true_norm={nd['p_true_norm']:.4f}, p_fused={nd['p_fused']:.4f}, p_for_bin={p_for_bin:.4f}")
print(f"  bin_mapper edges: {np.round(bin_mapper.get_edges(), 3)}")
print()

# Step 4: Training
print("--- Step 4: Training (1 epoch) ---")
bin_digit_tokens = get_bin_digit_tokens(tokenizer)
train_model.train()
optimizer.zero_grad()
valid_samples = 0
epoch_losses = []

for j, nd in enumerate(neighbor_data):
    p_for_bin = nd['p_true_norm']
    if nd['difficulty'] == 'input':
        bin_mapper.add(p_for_bin)
    target_bin = bin_mapper.to_bin(p_for_bin)
    nd['target_bin'] = target_bin
    print(f"  [{j}] ({nd['difficulty']}) target_bin={target_bin} (p_for_bin={p_for_bin:.4f})")
    
    prompt = create_qa_prompt(nd['question'])
    messages = [{"role": "user", "content": prompt}]
    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
    
    with torch.no_grad():
        outputs = train_model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
    
    # Log the training response
    train_response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    train_answer = extract_answer(train_response)
    train_conf = extract_confidence(train_response)
    print(f"    Training response: {train_response[:250]}")
    print(f"    Training answer: {train_answer!r}, conf: {train_conf}")
    
    response_tokens = outputs[:, inputs['input_ids'].shape[1]:].clone()
    conf_mask = create_confidence_mask(response_tokens, tokenizer)
    
    if conf_mask.sum() == 0:
        print(f"    !! No confidence mask found — skipping")
        del outputs
        continue
    
    # Log mask position and what token is there vs what we're replacing with
    mask_positions = (conf_mask[0] == 1).nonzero(as_tuple=True)[0]
    for pos in mask_positions:
        orig_token_id = response_tokens[0, pos].item()
        orig_token = tokenizer.decode([orig_token_id])
        target_token = tokenizer.decode([bin_digit_tokens[target_bin]])
        print(f"    Mask @ position {pos.item()}: original='{orig_token}' (bin{train_conf}) → target='{target_token}' (bin{target_bin})")
    
    labels = outputs.clone()
    labels[:, :inputs['input_ids'].shape[1]] = -100
    response_labels = labels[:, inputs['input_ids'].shape[1]:]
    target_token_id = bin_digit_tokens[target_bin]
    
    for i in range(response_tokens.shape[1]):
        if conf_mask[0, i] == 1:
            response_labels[0, i] = target_token_id
        else:
            response_labels[0, i] = -100
    labels[:, inputs['input_ids'].shape[1]:] = response_labels
    
    # Count how many tokens are being trained on (should be 1 — just the bin digit)
    trainable_tokens = (labels[0] != -100).sum().item()
    print(f"    Trainable tokens in label: {trainable_tokens} (expect ~1 for confidence digit only)")
    
    model_output = train_model(input_ids=outputs, attention_mask=torch.ones_like(outputs), labels=labels)
    loss = model_output.loss
    if loss is not None and not torch.isnan(loss):
        loss.backward()
        valid_samples += 1
        epoch_losses.append(loss.item())
        print(f"    loss={loss.item():.4f}")
    del outputs

if valid_samples > 0:
    for param in train_model.parameters():
        if param.grad is not None:
            param.grad.div_(valid_samples)
    torch.nn.utils.clip_grad_norm_(train_model.parameters(), max_norm=1.0)
    optimizer.step()
print(f"  Valid samples: {valid_samples}, Avg loss: {np.mean(epoch_losses) if epoch_losses else 0:.4f}")
print()

# Step 5: Post-TTT inference
print("--- Step 5: Post-TTT inference ---")
train_model.eval()
prompt = create_qa_prompt(question)
messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(train_model.device)
with torch.no_grad():
    outputs = train_model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.eos_token_id)
response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
del outputs
final_answer = extract_answer(response)
final_conf = extract_confidence(response)
with torch.no_grad():
    final_p_true = get_discriminative_confidence(question, final_answer, response, train_model, tokenizer,
                                                  use_p_know=False, claimed_bin=final_conf)
ground_truth = qa['ground_truth_answer']
is_correct = answers_match(final_answer, qa, dataset_type='mmlu')
print(f"  Final answer: {final_answer!r}")
print(f"  Final confidence: bin{final_conf}")
print(f"  Final P(True): {final_p_true:.4f}")
print(f"  Ground truth: {ground_truth}")
print(f"  Correct: {is_correct}")
print()

# Clean up LoRA
train_model.merge_and_unload()
del train_model, optimizer
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
# TEST 3: GSM8K (no mc_options — must generate distractors)
# ═══════════════════════════════════════════════════════════════════
qa_gsm = gsm_qs[0]
question_gsm = qa_gsm['question']

print("=" * 70)
print("  TEST 3: GSM8K (no MCQ options — distractors generated)")
print("=" * 70)
print(f"  Question: {question_gsm[:120]}...")
print(f"  mc_options: {qa_gsm.get('mc_options')}")
print()

baseline_gsm = get_baseline_answer(question_gsm, model, tokenizer)
print(f"  Answer: {baseline_gsm['answer']!r}")
print(f"  _is_mc_answer: {_is_mc_answer(baseline_gsm['answer'])}")

# No mc_options → must generate distractors
distractors = generate_distractors(question_gsm, baseline_gsm['answer'], model, tokenizer, k=4, mc_options=None)
candidates_gsm = [baseline_gsm['answer']] + distractors
print(f"  Candidates: {candidates_gsm}")

with torch.no_grad():
    p_scores_gsm = []
    for cand in candidates_gsm:
        p = get_discriminative_confidence(question_gsm, cand, baseline_gsm['response'], model, tokenizer,
                                          use_p_know=False, candidate_list=candidates_gsm)
        p_scores_gsm.append(p)
p_true_norm_gsm = normalize_p_true_sharpened(p_scores_gsm, temperature=args.norm_temperature)
print(f"  P(True) scores: {[f'{s:.3f}' for s in p_scores_gsm]}")
print(f"  p_true_norm: {p_true_norm_gsm:.4f}")
print()


# ═══════════════════════════════════════════════════════════════════
# TEST 4: Call train_single_question_discriminative DIRECTLY
# ═══════════════════════════════════════════════════════════════════
qa = mmlu_qs[2]
question = qa['question']

print("=" * 70)
print("  TEST 4: train_single_question_discriminative() — DIRECT CALL")
print("=" * 70)
print(f"  Question: {question[:120]}...")
print(f"  mc_options: {list(qa.get('mc_options', {}).keys())}")
print()

# Fresh LoRA
target_modules = R.get_late_layers_target_modules(args.layer_start, args.layer_end)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, r=args.lora_r, lora_alpha=args.lora_alpha,
    target_modules=target_modules, lora_dropout=0.05, bias="none",
)
train_model2 = get_peft_model(model, lora_config)
optimizer2 = torch.optim.AdamW(train_model2.parameters(), lr=args.lr)

result = train_single_question_discriminative(
    question, train_model2, optimizer2, tokenizer,
    n_neighbors=2, n_epochs=1, dataset_type='mmlu',
    neighbors_only=False, mc_options=qa.get('mc_options')
)

print(f"  TTT answer: {result['answer']!r}")
print(f"  TTT confidence: bin{result['confidence']}")
print(f"  TTT P(True): {result['p_true']:.4f}")
print(f"  Training losses: {[f'{l:.4f}' for l in result['training_losses']]}")
print(f"  Neighbors trained: {len(result['neighbors'])}")
for j, n in enumerate(result['neighbors']):
    print(f"    [{j}] ({n['difficulty']}) ans={n['answer']!r} p_true_norm={n['p_true_norm']:.4f}")
    # Check: did neighbors use MCQ letters?
    letter = _is_mc_answer(n['answer'])
    if n['difficulty'] != 'input':
        if letter:
            print(f"        ✓ Neighbor answered with MCQ letter '{letter}'")
        else:
            print(f"        ✗ Neighbor answered with TEXT '{n['answer']}' (not a letter)")
print()

# Check for any candidate duplication in the training function
gt = qa['ground_truth_answer']
is_correct = answers_match(result['answer'], qa, dataset_type='mmlu')
print(f"  Ground truth: {gt}")
print(f"  Correct: {is_correct}")

# Cleanup
train_model2.merge_and_unload()
del train_model2, optimizer2
torch.cuda.empty_cache()


# ═══════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  ALL TESTS COMPLETE")
print("=" * 70)
