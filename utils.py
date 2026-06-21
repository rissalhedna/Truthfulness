"""Utility functions for dataset loading, prompt creation, answer matching,
and calibration evaluation used by the SECL pipeline."""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter
from datetime import datetime
import random
import re
import time
from tqdm.auto import tqdm


def load_qa_dataset(dataset_mode="triviaqa", num_samples=100, model=None, tokenizer=None):
    """Load Q&A dataset for the specified benchmark.

    Args:
        dataset_mode: One of 'triviaqa', 'gsm8k', 'hle', 'truthfulqa',
                      'truthfulqa_mc', 'mmlu', or 'arc'
        num_samples: Number of Q&A pairs to load
        model: Unused (kept for API compatibility)
        tokenizer: Unused (kept for API compatibility)

    Returns:
        List of dicts with 'question', 'ground_truth_answer', and optionally
        'answer_aliases', 'mc_options', etc.
    """
    if dataset_mode == "triviaqa":
        print("Loading TriviaQA dataset...")
        trivia_dataset = load_dataset("trivia_qa", "unfiltered.nocontext", split="train")

        print("Extracting Q&A pairs from TriviaQA...")
        qa_pairs = []
        for i, example in enumerate(trivia_dataset):
            if len(qa_pairs) >= num_samples:
                break

            question = example['question'].strip()
            # Store all answer aliases for flexible matching
            answer_aliases = [ans.strip() for ans in example['answer']['aliases']] if example['answer']['aliases'] else [example['answer']['value'].strip()]
            primary_answer = answer_aliases[0]

            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")

            qa_pairs.append({
                'question': question,
                'ground_truth_answer': primary_answer,
                'answer_aliases': answer_aliases,  # Multiple correct answers
                'query_time': query_time_str,
                'passage': f"Trivia question: {question}"  # Placeholder
            })

        print(f"Loaded {len(qa_pairs)} Q&A pairs from TriviaQA")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question']}")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']} (+ {len(qa_pairs[0]['answer_aliases'])-1} aliases)")

    elif dataset_mode == "gsm8k":
        print("Loading GSM8K dataset...")
        gsm8k_dataset = load_dataset("openai/gsm8k", "main", split="train")

        print("Extracting Q&A pairs from GSM8K...")
        qa_pairs = []
        for i, example in enumerate(gsm8k_dataset):
            if len(qa_pairs) >= num_samples:
                break

            question = example['question'].strip()
            # Extract numerical answer from "#### <answer>" format
            answer_text = example['answer'].strip()
            answer_match = re.search(r'####\s*(.+)$', answer_text, re.MULTILINE)
            if answer_match:
                answer = answer_match.group(1).strip()
            else:
                # Fallback: use last line if #### not found
                answer = answer_text.split('\n')[-1].strip()

            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")

            qa_pairs.append({
                'question': question,
                'ground_truth_answer': answer,
                'answer_aliases': [answer],
                'query_time': query_time_str,
                'passage': f"Math problem: {question}"
            })

        print(f"Loaded {len(qa_pairs)} Q&A pairs from GSM8K")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question']}")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")

    elif dataset_mode == "hle":
        print("Loading HLE dataset...")
        hle_dataset = load_dataset("cais/hle", split="test")
        
        print("Filtering for text-only short-answer questions...")
        qa_pairs = []
        skipped_long = 0
        skipped_image = 0
        
        for example in hle_dataset:
            # Skip multimodal questions (with images)
            if 'image' in example and example['image'] is not None and example['image'] != '':
                skipped_image += 1
                continue
            
            # Filter for short answers only (<=10 words)
            answer = str(example['answer']).strip()
            if len(answer.split()) > 10:
                skipped_long += 1
                continue
            
            if len(qa_pairs) >= num_samples:
                break
            
            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")
            
            qa_pairs.append({
                'question': example['question'].strip(),
                'ground_truth_answer': answer,
                'answer_aliases': [answer],
                'query_time': query_time_str,
                'passage': f"HLE question: {example['question'][:200]}"
            })
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs from HLE")
        print(f"Skipped: {skipped_long} long answers, {skipped_image} multimodal")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question'][:100]}...")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")

    elif dataset_mode == "truthfulqa":
        print("Loading TruthfulQA dataset...")
        tqa_dataset = load_dataset("truthful_qa", "generation", split="validation")
        
        qa_pairs = []
        for example in tqa_dataset:
            if len(qa_pairs) >= num_samples:
                break
            
            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")
            
            qa_pairs.append({
                'question': example['question'].strip(),
                'ground_truth_answer': example['best_answer'].strip(),
                'answer_aliases': example['correct_answers'],  # List of all correct answers
                'query_time': query_time_str,
                'passage': f"TruthfulQA ({example['category']}): {example['question']}"
            })
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs from TruthfulQA")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question']}")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")
    
    elif dataset_mode == "truthfulqa_mc":
        print("Loading TruthfulQA multiple-choice dataset (MC1)...")
        tqa_dataset = load_dataset("truthful_qa", "multiple_choice", split="validation")
        
        qa_pairs = []
        for example in tqa_dataset:
            if len(qa_pairs) >= num_samples:
                break
            
            question = example["question"].strip()
            mc = example["mc1_targets"]
            choices = mc["choices"]
            labels = mc["labels"]
            
            # Map choices to letters
            letters = ["A", "B", "C", "D", "E", "F", "G", "H"]  # more than enough
            option_text = []
            mc_options = {}
            correct_letter = None
            for idx, choice in enumerate(choices):
                if idx >= len(letters):
                    break
                letter = letters[idx]
                option_text.append(f"{letter}. {choice}")
                mc_options[letter] = choice
                if labels[idx] == 1 and correct_letter is None:
                    correct_letter = letter
            
            if correct_letter is None:
                continue  # skip malformed
            
            # Build MC-formatted question
            mc_question = question + "\n" + "\n".join(option_text)
            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")
            
            qa_pairs.append({
                'question': mc_question,
                'ground_truth_answer': correct_letter,
                'answer_aliases': [correct_letter],
                'mc_options': mc_options,
                'query_time': query_time_str,
                'passage': f"TruthfulQA-MC: {question}"
            })
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs from TruthfulQA-MC")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question'].splitlines()[0]}")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")

    elif dataset_mode == "mmlu":
        print("Loading MMLU dataset (all subjects)...")
        # Load a diverse set of MMLU subjects
        subjects = [
            'abstract_algebra', 'anatomy', 'astronomy', 'business_ethics',
            'clinical_knowledge', 'college_biology', 'college_chemistry',
            'college_computer_science', 'college_mathematics', 'college_physics',
            'computer_security', 'conceptual_physics', 'econometrics',
            'electrical_engineering', 'elementary_mathematics', 'formal_logic',
            'global_facts', 'high_school_biology', 'high_school_chemistry',
            'high_school_computer_science', 'high_school_european_history',
            'high_school_geography', 'high_school_government_and_politics',
            'high_school_macroeconomics', 'high_school_mathematics',
            'high_school_microeconomics', 'high_school_physics',
            'high_school_psychology', 'high_school_statistics',
            'high_school_us_history', 'high_school_world_history',
            'human_aging', 'human_sexuality', 'international_law',
            'jurisprudence', 'logical_fallacies', 'machine_learning',
            'management', 'marketing', 'medical_genetics', 'miscellaneous',
            'moral_disputes', 'moral_scenarios', 'nutrition', 'philosophy',
            'prehistory', 'professional_accounting', 'professional_law',
            'professional_medicine', 'professional_psychology', 'public_relations',
            'security_studies', 'sociology', 'us_foreign_policy', 'virology',
            'world_religions'
        ]
        
        qa_pairs = []
        answer_map = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
        
        subject_iters = {}
        for subject in subjects:
            for attempt in range(5):  # Retry up to 5 times
                try:
                    try:
                        subject_iters[subject] = iter(load_dataset("cais/mmlu", subject, split="test", revision="main"))
                    except Exception as rev_e:
                        if "404" in str(rev_e) or "Entry Not Found" in str(rev_e):
                            # Stale cache or removed revision; try default branch
                            subject_iters[subject] = iter(load_dataset("cais/mmlu", subject, split="test"))
                        else:
                            raise
                    break
                except Exception as e:
                    if "429" in str(e) and attempt < 4:
                        wait = 2 ** attempt  # Exponential backoff: 1, 2, 4, 8 seconds
                        print(f"Rate limited on {subject}, retrying in {wait}s...")
                        time.sleep(wait)
                    else:
                        print(f"Warning: Could not load MMLU subject {subject}: {e}")
                        break
        
        # Round-robin through subjects
        while len(qa_pairs) < num_samples and subject_iters:
            exhausted = []
            for subject, data_iter in subject_iters.items():
                if len(qa_pairs) >= num_samples:
                    break
                try:
                    example = next(data_iter)
                    mc_options = {chr(65+i): c for i, c in enumerate(example['choices'])}
                    choices_text = "\n".join([f"{ltr}. {mc_options[ltr]}" for ltr in sorted(mc_options)])
                    full_question = f"{example['question']}\n\n{choices_text}"
                    query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")
                    
                    qa_pairs.append({
                        'question': full_question,
                        'ground_truth_answer': answer_map[example['answer']],
                        'answer_aliases': [answer_map[example['answer']]],
                        'mc_options': mc_options,
                        'query_time': query_time_str,
                        'passage': f"MMLU ({subject}): {example['question'][:100]}"
                    })
                except StopIteration:
                    exhausted.append(subject)
            
            for s in exhausted:
                del subject_iters[s]
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs from MMLU")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question'][:100]}...")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")

    elif dataset_mode == "arc":
        print("Loading ARC-Challenge dataset...")
        arc_dataset = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        
        qa_pairs = []
        for example in arc_dataset:
            if len(qa_pairs) >= num_samples:
                break
            
            # Format question with choices
            choices = example['choices']
            mc_options = {label: text for label, text in zip(choices['label'], choices['text'])}
            choices_text = "\n".join([f"{ltr}. {mc_options[ltr]}" for ltr in sorted(mc_options)])
            full_question = f"{example['question']}\n\n{choices_text}"
            
            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")
            
            qa_pairs.append({
                'question': full_question,
                'ground_truth_answer': example['answerKey'],
                'answer_aliases': [example['answerKey']],
                'mc_options': mc_options,
                'query_time': query_time_str,
                'passage': f"ARC: {example['question'][:100]}"
            })
        
        print(f"Loaded {len(qa_pairs)} Q&A pairs from ARC-Challenge")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question'][:100]}...")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")

    else:
        raise ValueError(f"Unknown dataset_mode: {dataset_mode}. Use 'triviaqa', 'gsm8k', 'hle', 'truthfulqa', 'mmlu', or 'arc'")

    return qa_pairs

DEFAULT_NUM_BINS = 10


def _confidence_bullets(num_bins=DEFAULT_NUM_BINS) -> str:
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    lines = []
    for i in range(num_bins):
        low = int(bin_edges[i] * 100)
        high = int(bin_edges[i + 1] * 100)
        lines.append(f"   - bin{i}. - you are ~{low}-{high}% confident your final answer is correct")
    return "\n".join(lines)


def create_qa_prompt(question, num_bins=DEFAULT_NUM_BINS):
    """Create directive prompt with confidence levels and structured reasoning"""
    return f"""Your task is to answer the question based on factual information in your own knowledge.

Please adhere to the following guidelines when formulating the answer:
You must rate how confident you are that your final answer is correct using one of these categories:
{_confidence_bullets(num_bins)}

IMPORTANT: Keep your reasoning BRIEF (1-2 sentences maximum). Then provide the final answer with your confidence level. The reasoning process must be enclosed within <think> </think> tags.
Do not add anything after the final answer.
Format:
<think>Your brief reasoning here (1-2 sentences max)</think>
Answer: <your answer>
Confidence: <confidence level>

For example:
Question: What is the capital of France?
<think>Paris is the capital of France.</think>
Answer: Paris.
Confidence: bin{num_bins - 1}.

Now answer the following question:
Question: {question}"""


def extract_answer(text):
    """Extract answer from model response with 'Answer: <answer>' format"""
    # Try to find "Answer:" explicitly labeled
    answer_match = re.search(r'Answer:\s*(.+?)(?=\n|Confidence:|$)', text, re.IGNORECASE)
    if answer_match:
        ans = answer_match.group(1).strip()
        if ans:
            return ans

    # Fallback: Get text after </think> tag but before Confidence:
    if '</think>' in text:
        after_think = text.split('</think>')[-1].strip()
        # Remove confidence line if present
        if 'Confidence:' in after_think:
            after_think = after_think.split('Confidence:')[0].strip()
        # Remove "Answer:" prefix if present
        if after_think.lower().startswith('answer:'):
            after_think = after_think[7:].strip()
        # Get first non-empty line
        lines = [l.strip() for l in after_think.split('\n') if l.strip()]
        if lines:
            return lines[0]

    # Last fallback: first non-empty line (excluding tags and confidence)
    lines = [l.strip() for l in text.split('\n') if l.strip() and not l.strip().startswith('<')]
    for line in lines:
        if 'confidence:' not in line.lower():
            return line

    # Final fallback: return full text (don't truncate for debugging)
    return text.strip()

def extract_confidence(text, num_bins=DEFAULT_NUM_BINS):
    """Extract verbalized confidence level from model response."""
    conf_match = re.search(r'Confidence:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
    if conf_match:
        conf_text = conf_match.group(1).strip().lower()
    else:
        return num_bins // 2

    for i in range(num_bins):
        if f"bin{i}" in conf_text:
            return i

    digit_match = re.search(r'^(\d+)', conf_text)
    if digit_match:
        digit = int(digit_match.group(1))
        if 0 <= digit < num_bins:
            return digit

    return num_bins // 2

def normalize_answer(text):
    """Normalize answer for comparison"""
    # Handle dict case (if full item is passed accidentally)
    if isinstance(text, dict):
        text = text.get('ground_truth_answer', '')
    if text is None:
        return ""
    # Convert to string and normalize
    text = str(text).lower().strip()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text

def extract_numerical_answer(text):
    """Extract numerical value from text (for GSM8K and numerical answers)
    
    Handles:
    - Plain numbers: 42, -3.14, 1,000
    - Mathematical expressions: 12 * (50/60), 10 + 5, etc.
    """
    if text is None:
        return None
    text = str(text).strip()
    
    # Strategy 1: Try to evaluate mathematical expressions
    # Look for expression patterns (numbers with operators)
    expr_pattern = r'[-\d\s+\-*/().,%]+'
    potential_exprs = re.findall(expr_pattern, text)
    
    for expr in reversed(potential_exprs):  # Start from the last expression
        expr = expr.strip().replace(',', '')  # Remove commas
        if not expr or expr.isspace():
            continue
        
        # Validate: only allow digits, operators, parentheses, spaces, and decimal points
        if not re.match(r'^[\d\s+\-*/().]+$', expr):
            continue
            
        # Check if it contains operators (making it an expression)
        if any(op in expr for op in ['+', '*', '/', '(', ')']):
            try:
                # Safely evaluate the expression
                # Restrict to safe namespace (only math operations)
                result = eval(expr, {"__builtins__": {}}, {})
                if isinstance(result, (int, float)):
                    if isinstance(result, float) and result.is_integer():
                        return int(result)
                    return result
            except Exception:
                # Catch all exceptions from eval - it's unpredictable
                continue
    
    # Strategy 2: Extract plain numbers (no expressions)
    # Match patterns like: 42, -42, 3.14, -3.14, 1,000, etc.
    number_pattern = r'-?\d+(?:,\d{3})*(?:\.\d+)?'
    matches = re.findall(number_pattern, text)
    if matches:
        # Take the last number (often the final answer in GSM8K)
        last_num = matches[-1].replace(',', '')
        try:
            # Try parsing as float first, then int if it's a whole number
            num = float(last_num)
            if num.is_integer():
                return int(num)
            return num
        except ValueError:
            return None
    return None

def answers_match(model_answer, ground_truth, threshold=0.7, dataset_type='auto'):
    """
    Check if model answer matches ground truth
    
    Matching strategies:
    - 'gsm8k': Numerical exact matching with expression evaluation
    - 'triviaqa': Text fuzzy matching with aliases and token overlap
    - 'auto': Try numerical first, fallback to text (default)

    Args:
        model_answer: The answer from the model (string)
        ground_truth: Either a string or a dict with 'ground_truth_answer' and optionally 'answer_aliases'
        threshold: Token overlap threshold for fuzzy text matching (default 0.7)
        dataset_type: 'gsm8k', 'triviaqa', or 'auto' (default 'auto')

    Returns:
        True if answers match, False otherwise
    """
    # Handle dict format (for compatibility with TriviaQA format)
    if isinstance(ground_truth, dict):
        gt_answers = ground_truth.get('answer_aliases', [ground_truth['ground_truth_answer']])
    elif isinstance(ground_truth, list):
        gt_answers = ground_truth
    else:
        gt_answers = [ground_truth]

    for gt_answer in gt_answers:
        # GSM8K: Numerical matching only
        if dataset_type == 'gsm8k':
            num_model = extract_numerical_answer(model_answer)
            num_gt = extract_numerical_answer(gt_answer)
            
            if num_model is not None and num_gt is not None:
                if num_model == num_gt:
                    return True
            # Continue to next alias if numbers don't match
            continue
        
        # TriviaQA: Text-based matching only
        elif dataset_type == 'triviaqa':
            norm_model = normalize_answer(model_answer)
            norm_gt = normalize_answer(gt_answer)
            
            # Exact match
            if norm_model == norm_gt:
                return True
            
            # Substring match
            if norm_model and norm_gt:
                if norm_model in norm_gt or norm_gt in norm_model:
                    return True
            
            # Token overlap
            tokens_model = set(norm_model.split())
            tokens_gt = set(norm_gt.split())
            if tokens_model and tokens_gt:
                overlap = len(tokens_model & tokens_gt) / max(len(tokens_model), len(tokens_gt))
                if overlap >= threshold:
                    return True
        
        # HLE: Exact text matching (normalized)
        elif dataset_type == 'hle':
            # HLE uses strict normalized exact match only
            return normalize_answer(model_answer) == normalize_answer(gt_answer)
        
        # TruthfulQA: Check against all correct answers (fuzzy)
        elif dataset_type == 'truthfulqa':
            norm_model = normalize_answer(model_answer)
            norm_gt = normalize_answer(gt_answer)
            
            # Exact match
            if norm_model == norm_gt:
                return True
            
            # Use ROUGE-L F1 score for fuzzy matching
            def lcs_length(x, y):
                """Longest common subsequence length."""
                m, n = len(x), len(y)
                dp = [[0] * (n + 1) for _ in range(m + 1)]
                for i in range(1, m + 1):
                    for j in range(1, n + 1):
                        if x[i-1] == y[j-1]:
                            dp[i][j] = dp[i-1][j-1] + 1
                        else:
                            dp[i][j] = max(dp[i-1][j], dp[i][j-1])
                return dp[m][n]
            
            tokens_model = norm_model.split()
            tokens_gt = norm_gt.split()
            
            if tokens_model and tokens_gt:
                lcs = lcs_length(tokens_model, tokens_gt)
                precision = lcs / len(tokens_model) if tokens_model else 0
                recall = lcs / len(tokens_gt) if tokens_gt else 0
                if precision + recall > 0:
                    rouge_l = 2 * precision * recall / (precision + recall)
                    if rouge_l >= 0.3:
                        return True
        
        # MMLU / TruthfulQA-MC: Multiple choice letter matching
        elif dataset_type in ['mmlu', 'truthfulqa_mc']:
            model_clean = model_answer.strip().upper()
            gt_letter = gt_answer.strip().upper()
            
            # Try to extract letter answer using patterns (strict matching)
            # Pattern 1: Just the letter alone
            if model_clean in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']:
                return model_clean == gt_letter
            
            # Pattern 2: "The answer is X", "Answer: X", "(X)", "X.", "X)"
            letter_patterns = [
                r'\b(?:answer\s*(?:is)?|the\s+answer\s*(?:is)?)\s*[:]*\s*([A-H])\b',
                r'\(([A-H])\)',
                r'^([A-H])[.)\s]',
                r'\b([A-H])\s*$',
            ]
            for pattern in letter_patterns:
                match = re.search(pattern, model_clean, re.IGNORECASE)
                if match:
                    return match.group(1).upper() == gt_letter
            
            # Fallback: reverse-map option text to letter via mc_options
            if isinstance(ground_truth, dict) and 'mc_options' in ground_truth:
                norm_model = normalize_answer(model_answer)
                for letter, option_text in ground_truth['mc_options'].items():
                    if normalize_answer(option_text) == norm_model:
                        return letter.upper() == gt_letter
            
            return False
        
        # ARC: Multiple choice letter matching
        elif dataset_type == 'arc':
            model_clean = model_answer.strip().upper()
            gt_letter = gt_answer.strip().upper()
            
            # Try to extract letter answer using patterns (strict matching)
            # Pattern 1: Just the letter alone
            if model_clean in ['A', 'B', 'C', 'D', 'E']:
                return model_clean == gt_letter
            
            # Pattern 2: "The answer is X", "Answer: X", "(X)", "X.", "X)"
            letter_patterns = [
                r'\b(?:answer\s*(?:is)?|the\s+answer\s*(?:is)?)\s*[:]*\s*([A-E])\b',
                r'\(([A-E])\)',
                r'^([A-E])[.)\s]',
                r'\b([A-E])\s*$',
            ]
            for pattern in letter_patterns:
                match = re.search(pattern, model_clean, re.IGNORECASE)
                if match:
                    return match.group(1).upper() == gt_letter
            
            # Fallback: reverse-map option text to letter via mc_options
            if isinstance(ground_truth, dict) and 'mc_options' in ground_truth:
                norm_model = normalize_answer(model_answer)
                for letter, option_text in ground_truth['mc_options'].items():
                    if normalize_answer(option_text) == norm_model:
                        return letter.upper() == gt_letter
            
            return False
        
        # Auto: Try numerical first, fallback to text
        else:  # dataset_type == 'auto'
            # Strategy 1: Try numerical matching first (for GSM8K)
            num_model = extract_numerical_answer(model_answer)
            num_gt = extract_numerical_answer(gt_answer)
            
            if num_model is not None and num_gt is not None:
                # Both are numbers - compare numerically
                if num_model == num_gt:
                    return True
                # Continue to next alias if numbers don't match
                continue
            
            # Strategy 2: Text-based matching (for TriviaQA and non-numerical answers)
            norm_model = normalize_answer(model_answer)
            norm_gt = normalize_answer(gt_answer)
            
            # Exact match after normalization
            if norm_model == norm_gt:
                return True
            
            # Substring match (only if both are non-empty and neither is purely numerical)
            if norm_model and norm_gt and num_model is None and num_gt is None:
                if norm_model in norm_gt or norm_gt in norm_model:
                    return True
            
            # Token overlap
            tokens_model = set(norm_model.split())
            tokens_gt = set(norm_gt.split())
            if tokens_model and tokens_gt:
                overlap = len(tokens_model & tokens_gt) / max(len(tokens_model), len(tokens_gt))
                if overlap >= threshold:
                    return True

    return False
