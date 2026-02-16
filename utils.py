import torch
import numpy as np
import pandas as pd
from datasets import load_dataset
from collections import Counter
from datetime import datetime
import random
import re
from tqdm.auto import tqdm

torch.manual_seed(42)
np.random.seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")


def generate_qa_pairs(passages, model, tokenizer, num_pairs=50):
    """Generate question-answer pairs from passages with current timestamp"""
    qa_pairs = []

    for passage in tqdm(passages[:num_pairs], desc="Generating Q&A pairs"):
        # Prompt to generate a question and answer from the passage
        prompt = f"""Based on the following passage, generate one factual question and its answer.

Passage: {passage[:500]}

Generate:
Question: <your question here>
Answer: <your answer here>

Remember: The answer must be directly supported by the passage.""".strip()

        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        del outputs  # Free memory

        # Parse question and answer
        question_match = re.search(r'Question:\s*(.+?)(?=\nAnswer:|\n\n|$)', generated, re.DOTALL)
        answer_match = re.search(r'Answer:\s*(.+?)(?=\n\n|$)', generated, re.DOTALL)

        if question_match and answer_match:
            question = question_match.group(1).strip()
            answer = answer_match.group(1).strip()

            # Use current timestamp for this question
            query_time_str = datetime.now().strftime("%m/%d/%Y, %H:%M:%S PT")

            qa_pairs.append({
                'passage': passage,
                'question': question,
                'ground_truth_answer': answer,
                'query_time': query_time_str
            })

    return qa_pairs
def load_qa_dataset(dataset_mode="triviaqa", num_samples=100, model=None, tokenizer=None):
    """
    Load Q&A dataset with three modes:
    - 'triviaqa': Use TriviaQA with multiple correct answers
    - 'wikitext': Generate Q&A pairs from WikiText passages using LLM
    - 'gsm8k': Use GSM8K math word problems

    Args:
        dataset_mode: "triviaqa", "wikitext", or "gsm8k"
        num_samples: Number of Q&A pairs to load/generate
        model: Required for 'wikitext' mode
        tokenizer: Required for 'wikitext' mode

    Returns:
        qa_pairs: List of Q&A dictionaries with 'question', 'ground_truth_answer',
                  'query_time', 'passage', and optionally 'answer_aliases'
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

    elif dataset_mode == "wikitext":
        if model is None or tokenizer is None:
            raise ValueError("model and tokenizer are required for 'wikitext' mode")

        print("Loading WikiText-103 dataset...")
        wiki_dataset = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

        # Extract passages
        print("Extracting passages from WikiText...")
        passages = []
        for i, example in enumerate(wiki_dataset):
            if len(passages) >= num_samples:
                break

            text = example['text'].strip()

            # Filter for good quality passages
            if (500 <= len(text) <= 2000 and
                text.count('.') >= 3 and
                text.count('\n') < 10):
                passages.append(text)

        print(f"Loaded {len(passages)} passages from WikiText")

        # Generate Q&A pairs using LLM
        print("Generating Q&A pairs from passages (this may take a while)...")
        qa_pairs = generate_qa_pairs(passages, model, tokenizer, num_pairs=len(passages))

        print(f"Generated {len(qa_pairs)} Q&A pairs from WikiText")
        if qa_pairs:
            print(f"Example: Q: {qa_pairs[0]['question']}")
            print(f"         A: {qa_pairs[0]['ground_truth_answer']}")

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
        
        # Round-robin sample from subjects until we reach num_samples
        # Add retry logic for rate limits
        import time as _time
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
                        _time.sleep(wait)
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


def create_math_qa_prompt(question, num_bins=DEFAULT_NUM_BINS):
    """Create directive prompt with confidence levels and structured reasoning"""
    return f"""Your task is to answer the question based on factual information in your own knowledge.

Please adhere to the following guidelines when formulating the answer:
You must rate how confident you are that your final answer is correct using one of these categories:
{_confidence_bullets(num_bins)}

IMPORTANT: Keep your reasoning BRIEF (1-2 sentences maximum). Then provide the final answer with your confidence level. The reasoning process must be enclosed within <think> </think> tags.
For math questions, the answer should be a final number with no units. Do not give an equation in your final answer, only final number. Do not add anything after the final answer.
Format:
<think>Your brief reasoning here (1-2 sentences max)</think>
Answer: <your answer>
Confidence: <confidence level>

For example:
Question: If you have 10 apples, and you eat two of them, but you buy seven more, how many apples do you have?
<think>You start with 10 apples, eat 2, buy 7. 10 - 2 + 7 = 15.</think>
Answer: 15.
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
    return text.strip()  # ✅ FIXED: Was [:100], now returns full text

confidence_extraction_stats = {'success': 0, 'fallback_no_match': 0, 'fallback_no_parse': 0}

def extract_confidence(text, num_bins=DEFAULT_NUM_BINS):
    """Extract verbalized confidence level from model response."""
    conf_match = re.search(r'Confidence:\s*(.+?)(?=\n|$)', text, re.IGNORECASE)
    if conf_match:
        conf_text = conf_match.group(1).strip().lower()
    else:
        confidence_extraction_stats['fallback_no_match'] += 1
        return num_bins // 2

    for i in range(num_bins):
        if f"bin{i}" in conf_text:
            confidence_extraction_stats['success'] += 1
            return i
    
    digit_match = re.search(r'^(\d+)', conf_text)
    if digit_match:
        digit = int(digit_match.group(1))
        if 0 <= digit < num_bins:
            confidence_extraction_stats['success'] += 1
            return digit

    confidence_extraction_stats['fallback_no_parse'] += 1
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
            if model_clean in ['A', 'B', 'C', 'D']:
                return model_clean == gt_letter
            
            # Pattern 2: "The answer is X", "Answer: X", "(X)", "X.", "X)"
            letter_patterns = [
                r'\b(?:answer\s*(?:is)?|the\s+answer\s*(?:is)?)\s*[:]*\s*([ABCD])\b',
                r'\(([ABCD])\)',
                r'^([ABCD])[.)\s]',
                r'\b([ABCD])\s*$',  # Letter at end of string
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
            if model_clean in ['A', 'B', 'C', 'D']:
                return model_clean == gt_letter
            
            # Pattern 2: "The answer is X", "Answer: X", "(X)", "X.", "X)"
            letter_patterns = [
                r'\b(?:answer\s*(?:is)?|the\s+answer\s*(?:is)?)\s*[:]*\s*([ABCD])\b',
                r'\(([ABCD])\)',
                r'^([ABCD])[.)\s]',
                r'\b([ABCD])\s*$',  # Letter at end of string
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

def compute_correctness_rates(qa_dataset, model, tokenizer, n_samples=10, 
                              max_questions=100, temperature=1.0, dataset_type='auto',
                              save_path="results/precomputed_correctness_rates.json"):
    """
    Pre-compute actual correctness rates for any Q&A dataset (TriviaQA, GSM8K, etc.)
    
    For each question:
    - Generate n_samples answers with temperature sampling
    - Check how many are correct vs ground truth
    - Calculate correctness_rate = num_correct / n_samples
    - Save results incrementally every 10 questions
    - Can resume from interruption (loads existing results and skips processed questions)
    
    Args:
        qa_dataset: List of question dicts (from TriviaQA, GSM8K, or custom datasets)
        model: The model to evaluate
        tokenizer: Tokenizer
        n_samples: Number of samples per question
        max_questions: Maximum questions to process
        temperature: Sampling temperature
        dataset_type: 'gsm8k', 'triviaqa', or 'auto' (default 'auto')
        save_path: Path to save JSON file (default: "results/precomputed_correctness_rates.json")
    
    Returns:
        dict: Mapping from question index (str) to correctness data dict
    """
    import json
    import os
    
    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Load existing results if file exists (resume capability)
    if os.path.exists(save_path):
        print(f"\n✓ Found existing file: {save_path}")
        print("Loading existing results to resume...")
        with open(save_path, 'r') as f:
            correctness_rates = json.load(f)
        print(f"✓ Loaded {len(correctness_rates)} existing results")
        
        # Check if already complete
        if len(correctness_rates) >= min(max_questions, len(qa_dataset)):
            print("✓ All questions already processed!")
            return correctness_rates
        
        print(f"→ Resuming from question {len(correctness_rates)}")
    else:
        correctness_rates = {}
    
    print(f"\n⏳ Computing correctness rates for {min(max_questions, len(qa_dataset))} questions...")
    print(f"Generating {n_samples} samples per question with temperature={temperature}")
    print(f"Dataset type: {dataset_type}")
    print(f"💾 Saving incrementally every 10 questions to: {save_path}")
    
    model.eval()
    
    for idx, qa in enumerate(tqdm(qa_dataset[:max_questions], desc="Computing correctness rates")):
        # Skip if already processed
        if str(idx) in correctness_rates:
            continue
            
        question = qa['question']
        ground_truth = qa['ground_truth_answer']
        aliases = qa.get('answer_aliases', [ground_truth])
        
        # Create prompt (use math prompt for GSM8K)
        if dataset_type == 'gsm8k':
            prompt = create_math_qa_prompt(question)
        else:
            prompt = create_qa_prompt(question)
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        # Generate multiple samples
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=temperature,
                do_sample=True,
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id
            )
        
        # Extract and check answers
        sampled_answers = []
        correct_count = 0
        
        for output in outputs:
            response = tokenizer.decode(output[inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            answer = extract_answer(response)
            sampled_answers.append(answer)
            
            # Check if correct
            is_correct = answers_match(answer, qa, dataset_type=dataset_type)
            if is_correct:
                correct_count += 1
        
        del outputs
        torch.cuda.empty_cache()
        
        # Calculate correctness rate
        correctness_rate = correct_count / n_samples
        
        # Get the most common answer
        answer_counts = Counter(sampled_answers)
        most_common_answer = answer_counts.most_common(1)[0][0]
        
        correctness_rates[str(idx)] = {
            'question': question,
            'ground_truth_answer': ground_truth,
            'answer_aliases': aliases,
            'sampled_answers': sampled_answers,
            'most_common_answer': most_common_answer,
            'num_correct': correct_count,
            'correctness_rate': correctness_rate,
            'n_samples': n_samples
        }
        
        # Save incrementally every 10 questions
        if (idx + 1) % 10 == 0:
            with open(save_path, 'w') as f:
                json.dump(correctness_rates, f, indent=2)
            avg_rate = sum(r['correctness_rate'] for r in correctness_rates.values()) / len(correctness_rates)
            print(f"\n  💾 Saved at {idx + 1}/{min(max_questions, len(qa_dataset))} | Avg correctness rate: {avg_rate:.3f}")
    
    torch.cuda.empty_cache()
    
    # Final save
    with open(save_path, 'w') as f:
        json.dump(correctness_rates, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETED! Final save: {save_path}")
    print(f"✓ Total questions processed: {len(correctness_rates)}")
    avg_correctness = sum(r['correctness_rate'] for r in correctness_rates.values()) / len(correctness_rates)
    print(f"✓ Average correctness rate: {avg_correctness:.3f}")
    print(f"{'='*60}")
    
    return correctness_rates


def evaluate_calibration(qa_dataset, sample_results, model, tokenizer, max_eval=20, dataset_type='auto'):
    """
    Evaluate model calibration on Q&A dataset using pre-computed sample results
    
    Works with ANY Q&A dataset (TriviaQA, GSM8K, etc.) - fully generic!
    - TriviaQA: Text answers matched with fuzzy matching + aliases
    - GSM8K: Numerical answers matched with expression evaluation
    
    Calibration Check:
    - Generate answer with temperature=0 to get stated confidence (verbalized_confidence_bin)
    - Use pre-computed correctness_rate (% of temperature=1 samples that were correct)
    - Compare: Does stated confidence match actual correctness rate?
    
    Args:
        qa_dataset: List of Q&A questions (TriviaQA, GSM8K, or any dataset with 
                    'question' and 'ground_truth_answer' fields)
        sample_results: List of dicts from compute_correctness_rates, each with:
            - correctness_rate: % of samples that were correct (actual performance)
            - sampled_answers: List of all sampled answers
        model: Model to generate verbalized confidence
        tokenizer: Tokenizer
        max_eval: Maximum number of questions to evaluate
        dataset_type: 'gsm8k', 'triviaqa', or 'auto' (default 'auto')
    
    Returns:
        List of dicts with:
            - verbalized_confidence_bin: Model's stated confidence (0-4)
            - correctness_rate: Actual % of samples correct (0.0-1.0)
            - Other metadata
    """
    results = []

    for i, (qa, sample_result) in enumerate(tqdm(zip(qa_dataset[:max_eval], sample_results[:max_eval]), 
                                                   desc="Evaluating calibration", 
                                                   total=min(max_eval, len(sample_results)))):
        # Generate answer with verbalized confidence (temperature=0 for consistency)
        if dataset_type == 'gsm8k':
            prompt = create_math_qa_prompt(qa['question'])
        else:
            prompt = create_qa_prompt(qa['question'])
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        verbalized_confidence_bin = extract_confidence(response)
        model_answer = extract_answer(response)
        
        del outputs
        torch.cuda.empty_cache()
        
        # Check if the temperature=0 answer is correct
        # answers_match() handles both text (TriviaQA) and numerical (GSM8K) answers
        is_correct = answers_match(model_answer, qa, dataset_type=dataset_type)
        
        results.append({
            'question': qa['question'],
            'ground_truth_answer': qa['ground_truth_answer'],
            'model_answer': model_answer,
            'is_correct': is_correct,
            'correctness_rate': sample_result['correctness_rate'],
            'verbalized_confidence_bin': verbalized_confidence_bin,
            'num_correct': sample_result['num_correct'],
            'n_samples': sample_result['n_samples'],
            'sampled_answers': sample_result['sampled_answers']
        })

    return results

print("✅ Calibration evaluation functions defined!")

def visualize_calibration(calib_results, title_prefix=""):
    """
    Visualize calibration results with comprehensive plots and summary statistics.
    
    Args:
        calib_results: Dictionary returned by calculate_calibration_metrics containing:
            - 'calibration_by_bin': List of per-bin calibration data
            - 'brier_score': Overall Brier score
            - 'ece': Expected Calibration Error
            - 'accuracy': Overall accuracy
        title_prefix: Optional prefix for plot titles (e.g., "Baseline" or "Trained")
    
    Returns:
        fig: Matplotlib figure object
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    
    # Set style
    sns.set_style("whitegrid")
    
    # Extract calibration data
    calib_by_bin = calib_results['calibration_by_bin']
    
    # Create a complete view with all 5 bins (fill missing bins with None/0)
    all_bins = list(range(5))
    bin_to_prob = {0: 0.1, 1: 0.3, 2: 0.5, 3: 0.7, 4: 0.9}
    
    # Build dictionaries for quick lookup
    bin_data_dict = {item['bin']: item for item in calib_by_bin}
    
    # Extract data for bins with data (for calibration curve plot)
    bins_with_data = [item['bin'] for item in calib_by_bin]
    stated_conf_with_data = [item['stated_confidence'] for item in calib_by_bin]
    actual_conf_with_data = [item['avg_correctness_rate'] for item in calib_by_bin]
    
    bin_labels = ['bin0\n(0-20%)', 'bin1\n(20-40%)', 'bin2\n(40-60%)', 'bin3\n(60-80%)', 'bin4\n(80-100%)']
    
    # For bar charts, we need all bins
    stated_conf_all = [bin_data_dict[i]['stated_confidence'] if i in bin_data_dict else bin_to_prob[i] for i in all_bins]
    actual_conf_all = [bin_data_dict[i]['avg_correctness_rate'] if i in bin_data_dict else 0 for i in all_bins]
    counts_all = [bin_data_dict[i]['count'] if i in bin_data_dict else 0 for i in all_bins]
    gaps_all = [bin_data_dict[i]['calibration_gap'] if i in bin_data_dict else 0 for i in all_bins]
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Calibration Curve (Reliability Diagram) - TOP LEFT
    ax1 = axes[0, 0]
    
    # Plot perfect calibration line (45-degree diagonal)
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration', alpha=0.7)
    
    # Plot actual calibration (only bins with data)
    ax1.plot(stated_conf_with_data, actual_conf_with_data, 'o-', linewidth=3, markersize=12, 
             color='steelblue', label='Model Calibration', alpha=0.8)
    
    # Add bin labels to points
    for i, (sc, ac, b) in enumerate(zip(stated_conf_with_data, actual_conf_with_data, bins_with_data)):
        ax1.annotate(f'bin{b}', (sc, ac), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10, fontweight='bold')
    
    # Show all expected bin positions (even if no data) as light gray dots
    for bin_idx in all_bins:
        if bin_idx not in bins_with_data:
            expected_conf = bin_to_prob[bin_idx]
            ax1.plot(expected_conf, expected_conf, 'o', markersize=8, color='lightgray', alpha=0.5)
            ax1.annotate(f'bin{bin_idx}\n(no data)', (expected_conf, expected_conf), 
                        textcoords="offset points", xytext=(0,10), ha='center', 
                        fontsize=8, color='gray', alpha=0.7)
    
    ax1.set_xlabel('Predicted Probability (Stated Confidence)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Actual Frequency (Correctness Rate)', fontsize=12, fontweight='bold')
    title = f'{title_prefix} Calibration Curve (Reliability Diagram)' if title_prefix else 'Calibration Curve (Reliability Diagram)'
    ax1.set_title(title, fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Add shaded region for well-calibrated area (±0.1)
    ax1.fill_between([0, 1], [0, 1], [0.1, 1.1], alpha=0.1, color='green', label='Well-calibrated (±0.1)')
    ax1.fill_between([0, 1], [-0.1, 0.9], [0, 1], alpha=0.1, color='green')
    
    # Plot 2: Stated vs Actual Confidence (Bar Chart) - TOP RIGHT
    ax2 = axes[0, 1]
    x = np.arange(len(all_bins))
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, stated_conf_all, width, label='Stated Confidence', alpha=0.8, color='steelblue')
    bars2 = ax2.bar(x + width/2, actual_conf_all, width, label='Actual Correctness Rate', alpha=0.8, color='coral')
    
    ax2.set_xlabel('Confidence Bin', fontsize=12)
    ax2.set_ylabel('Confidence / Correctness Rate', fontsize=12)
    title = f'{title_prefix} Stated vs Actual by Bin' if title_prefix else 'Stated vs Actual by Bin'
    ax2.set_title(title, fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(bin_labels)
    ax2.legend()
    ax2.set_ylim([0, 1.0])
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars (only for bins with data)
    for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
        if i in bin_data_dict:  # Only label bins with data
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            ax2.text(bar1.get_x() + bar1.get_width()/2., height1,
                    f'{height1:.2f}', ha='center', va='bottom', fontsize=9)
            if height2 > 0:  # Only show actual if > 0
                ax2.text(bar2.get_x() + bar2.get_width()/2., height2,
                        f'{height2:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Calibration Gap - BOTTOM LEFT
    ax3 = axes[1, 0]
    colors = ['green' if gap < 0.1 else 'orange' if gap < 0.2 else 'red' for gap in gaps_all]
    bars = ax3.bar(x, gaps_all, color=colors, alpha=0.7)
    
    ax3.set_xlabel('Confidence Bin', fontsize=12)
    ax3.set_ylabel('Calibration Gap (|Stated - Actual|)', fontsize=12)
    title = f'{title_prefix} Calibration Error by Bin' if title_prefix else 'Calibration Error by Bin'
    ax3.set_title(title, fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(bin_labels)
    ax3.axhline(y=0.1, color='green', linestyle='--', alpha=0.5, label='Good (<0.1)')
    ax3.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Fair (<0.2)')
    ax3.legend(loc='upper right')
    ax3.grid(axis='y', alpha=0.3)
    
    # Add value labels (only for bins with data)
    for i, bar in enumerate(bars):
        if i in bin_data_dict:
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9)
    
    # Plot 4: Sample Count Distribution - BOTTOM RIGHT
    ax4 = axes[1, 1]
    bars = ax4.bar(x, counts_all, color='mediumpurple', alpha=0.7)
    
    ax4.set_xlabel('Confidence Bin', fontsize=12)
    ax4.set_ylabel('Number of Questions', fontsize=12)
    title = f'{title_prefix} Question Distribution by Confidence Bin' if title_prefix else 'Question Distribution by Confidence Bin'
    ax4.set_title(title, fontsize=14, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(bin_labels)
    ax4.grid(axis='y', alpha=0.3)
    
    # Add value labels (only for bins with data)
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if height > 0:  # Only show label if there's data
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    prefix_str = f"{title_prefix} " if title_prefix else ""
    print("\n" + "="*80)
    print(f"{prefix_str}CALIBRATION SUMMARY")
    print("="*80)
    print(f"Overall Brier Score: {calib_results['brier_score']:.4f} (lower is better, 0 = perfect)")
    print(f"Overall ECE (Expected Calibration Error): {calib_results['ece']:.4f}")
    print(f"Overall Accuracy: {calib_results['accuracy']:.4f}")
    print(f"\nPer-Bin Breakdown:")
    print("-"*80)
    print(f"{'Bin':<8} {'Stated':<10} {'Actual':<10} {'Gap':<10} {'Count':<8}")
    print("-"*80)
    for item in calib_by_bin:
        print(f"bin{item['bin']:<5} {item['stated_confidence']:<10.3f} {item['avg_correctness_rate']:<10.3f} "
              f"{item['calibration_gap']:<10.3f} {item['count']:<8}")
    print("="*80)
    
    return fig



# ============================================================================
# IMPROVED Question Generation with Diversity & Quality Control
# ============================================================================

def generate_diverse_quality_questions(model, tokenizer, num_questions=300, 
                                       target_difficulty_distribution=None, 
                                       temperature=0.5, save = False, save_path = None):
    """
    Generate high-quality, diverse questions with controlled difficulty distribution
    
    Features:
    1. Quality filtering (grammatical, specific, answerable)
    2. Diversity enforcement (avoid repetitive topics)
    3. Difficulty stratification (easy, medium, hard questions)
    4. Balanced across domains
    
    Args:
        num_questions: Total questions to generate
        target_difficulty_distribution: Dict like {'easy': 0.3, 'medium': 0.4, 'hard': 0.3}
        temperature: Generation temperature
    
    Returns:
        List of high-quality diverse questions
    """
    
    if target_difficulty_distribution is None:
        target_difficulty_distribution = {'easy': 0.33, 'medium': 0.34, 'hard': 0.33}
    
    targets = {
        'easy': int(num_questions * target_difficulty_distribution['easy']),
        'medium': int(num_questions * target_difficulty_distribution['medium']),
        'hard': int(num_questions * target_difficulty_distribution['hard'])
    }
    
    domains = {
        'easy': [
            "basic geography", "famous people", "common animals", "everyday objects",
            "popular movies", "simple math", "basic science", "common foods"
        ],
        'medium': [
            "world history", "scientific concepts", "literature", "technology",
            "economics", "art movements", "political systems", "medical terms"
        ],
        'hard': [
            "advanced physics", "philosophy", "rare historical events", "specialized biology",
            "abstract mathematics", "classical music", "ancient civilizations", "legal concepts"
        ]
    }
    
    all_questions = []
    seen_topics = []
    
    for difficulty in ['easy', 'medium', 'hard']:
        print(f"\n{'='*60}")
        print(f"Generating {targets[difficulty]} {difficulty.upper()} questions...")
        print(f"{'='*60}")
        
        questions_for_difficulty = []
        attempts = 0
        max_attempts = targets[difficulty] * 5
        
        # Dynamic domain limit: scale based on target questions
        # For 2000 total questions: ~80 questions per domain
        # For 300 total questions: ~15 questions per domain
        max_per_domain = max(15, (targets[difficulty] // len(domains[difficulty])) * 2)
        print(f"  Max questions per domain: {max_per_domain}")
        
        while len(questions_for_difficulty) < targets[difficulty] and attempts < max_attempts:
            domain = domains[difficulty][attempts % len(domains[difficulty])]
            
            # Dynamic domain limit (scales with num_questions)
            if seen_topics.count(domain) >= max_per_domain:
                attempts += 1
                continue
            
            prompt = f"""Generate a single, high-quality {difficulty} difficulty question about {domain}.

            Requirements:
            - Must be SPECIFIC (not vague or ambiguous)
            - Must be ANSWERABLE with factual knowledge
            - Must be CLEAR and grammatically correct
            - Should have ONE correct answer (not opinion-based)
            - {difficulty.capitalize()} difficulty: {'Common knowledge' if difficulty == 'easy' else 'Specialized knowledge' if difficulty == 'hard' else 'General but not trivial'}

            Format:
            Question: <your question here>

            Example ({difficulty}):
            Question: {"What is the capital of France?" if difficulty == "easy" else "What year did the Treaty of Westphalia end the Thirty Years' War?" if difficulty == "hard" else "Who wrote the novel '1984'?"}

            Now generate ONE {difficulty} question about {domain}:
            """.strip()
            
            messages = [{"role": "user", "content": prompt}]
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=temperature,
                    do_sample=True,  # Enable sampling to get diverse questions
                    pad_token_id=tokenizer.eos_token_id,
                )
            
            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            del outputs
            
            if "Question:" in response:
                question = response.split("Question:")[1].split("\n")[0].strip()
            else:
                question = response.split("\n")[0].strip()
            
            if not question or len(question) < 10:
                attempts += 1
                continue
            
            # Normalize: remove trailing ?, strip whitespace, collapse multiple spaces
            question = question.rstrip('?').strip()
            question = ' '.join(question.split())  # Collapse multiple spaces into one
            
            # Normalize for duplicate checking
            question_normalized = question.lower().strip()
            
            quality_issues = []
            if len(question.split()) < 4:
                quality_issues.append("too short")
            if len(question.split()) > 50:
                quality_issues.append("too long")
            if not any(c.isalpha() for c in question):
                quality_issues.append("no letters")
            
            # Check for duplicates using normalized versions
            existing_questions = [q['question'].lower().strip() for q in all_questions]
            current_batch_questions = [q['question'].lower().strip() for q in questions_for_difficulty]
            
            if question_normalized in existing_questions or question_normalized in current_batch_questions:
                quality_issues.append("duplicate")
            
            if quality_issues:
                attempts += 1
                continue
            
            questions_for_difficulty.append({
                'question': question,
                'difficulty': difficulty,
                'domain': domain,
            })
            
            seen_topics.append(domain)
            attempts += 1
            
            if len(questions_for_difficulty) % 20 == 0:
                print(f"  ✓ Generated {len(questions_for_difficulty)}/{targets[difficulty]} {difficulty} questions")
        
        print(f"✅ Completed {len(questions_for_difficulty)} {difficulty} questions")
        all_questions.extend(questions_for_difficulty)
    
    random.shuffle(all_questions)
    
    print(f"\n{'='*60}")
    print(f"GENERATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions: {len(all_questions)}")
    print(f"  • Easy: {sum(1 for q in all_questions if q['difficulty'] == 'easy')}")
    print(f"  • Medium: {sum(1 for q in all_questions if q['difficulty'] == 'medium')}")
    print(f"  • Hard: {sum(1 for q in all_questions if q['difficulty'] == 'hard')}")
    print(f"Unique domains: {len(set(seen_topics))}")
    print(f"{'='*60}")
    
    if save:
        import json
        with open(save_path, 'w') as f:
            json.dump(all_questions, f, indent=2)
        print(f"✓ Saved {len(all_questions)} questions to {save_path}")
    
    return all_questions

def precompute_self_consistency_rates(qa_dataset, model, tokenizer, 
                                      n_samples=50, dataset_mode="gsm8k", save_path="results/precomputed_self_consistency.json",
                                      debug_freq=50):
    """
    Precompute self-consistency rates for self-generated questions with resumability
    
    For each question:
    - Generate n_samples answers with temperature 1.0
    - Calculate pairwise agreement rate (self-consistency)
    - Save results incrementally every 10 questions
    - Can resume from interruption (loads existing results and skips processed questions)
    
    Args:
        qa_dataset: List of dicts with 'question' key
        model: Language model to use for generation
        tokenizer: Tokenizer
        n_samples: Number of samples to generate per question (default: 50)
        dataset_mode: "gsm8k" or other dataset mode (default: "gsm8k")
        save_path: Path to save JSON file (default: "results/precomputed_self_consistency.json")
        debug_freq: Print debug info every N questions (default: 50, set to 0 to disable)
    
    Returns:
        dict: Mapping from question index (str) to consistency rate (float)
    """
    import json
    import os

    # Ensure save directory exists
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    
    # Load existing results if file exists (resume capability)
    if os.path.exists(save_path):
        print(f"\n✓ Found existing file: {save_path}")
        print("Loading existing results to resume...")
        with open(save_path, 'r') as f:
            self_consistency_rates = json.load(f)
        print(f"✓ Loaded {len(self_consistency_rates)} existing results")
        
        # Check if already complete
        if len(self_consistency_rates) >= len(qa_dataset):
            print("✓ All questions already processed!")
            return self_consistency_rates
        
        print(f"→ Resuming from question {len(self_consistency_rates)}")
    else:
        self_consistency_rates = {}
    
    print(f"\n⏳ Precomputing self-consistency rates for {len(qa_dataset)} questions...")
    print(f"Generating {n_samples} samples per question with temperature=1.0")
    print(f"💾 Saving incrementally every 10 questions to: {save_path}")
    if debug_freq > 0:
        print(f"🔍 Debug output enabled: showing details every {debug_freq} questions")
    
    model.eval()
    
    for idx, q_dict in enumerate(tqdm(qa_dataset, desc="Precomputing self-consistency")):
        # Skip if already processed
        if str(idx) in self_consistency_rates:
            continue
        
        question = q_dict['question']
        
        if dataset_mode == "gsm8k":
            prompt = create_math_qa_prompt(question)
        else:
            prompt = create_qa_prompt(question)
        messages = [{"role": "user", "content": prompt}]
        input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
        
        with torch.no_grad():
            sample_outputs = model.generate(
                **inputs, 
                max_new_tokens=200, 
                temperature=1.0,
                do_sample=True, 
                num_return_sequences=n_samples,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        sampled_answers = []
        for i in range(n_samples):
            sample_response = tokenizer.decode(sample_outputs[i][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            sample_answer = extract_answer(sample_response)
            sampled_answers.append(sample_answer)
        
        del sample_outputs
        
        if len(sampled_answers) > 1:
            total_pairs = 0
            agreement_pairs = 0
            for i in range(len(sampled_answers)):
                for j in range(i + 1, len(sampled_answers)):
                    total_pairs += 1
                    if answers_match(sampled_answers[i], {'ground_truth_answer': sampled_answers[j]}):
                        agreement_pairs += 1
            
            consistency_rate = agreement_pairs / total_pairs if total_pairs > 0 else 0.0
        else:
            consistency_rate = 1.0
        
        self_consistency_rates[str(idx)] = consistency_rate
        
        # Debug output
        if debug_freq > 0 and (idx + 1) % debug_freq == 0:
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG - Question {idx + 1}/{len(qa_dataset)}")
            print(f"{'='*80}")
            print(f"❓ Question: {question[:200]}..." if len(question) > 200 else f"❓ Question: {question}")
            print(f"\n📊 Sampled {len(sampled_answers)} answers:")
            # Show first 5 unique answers
            unique_answers = list(set(sampled_answers))[:5]
            for i, ans in enumerate(unique_answers, 1):
                count = sampled_answers.count(ans)
                ans_preview = ans[:100] + "..." if len(str(ans)) > 100 else ans
                print(f"  {i}. [{count}/{n_samples}] {ans_preview}")
            if len(unique_answers) < len(set(sampled_answers)):
                print(f"  ... and {len(set(sampled_answers)) - len(unique_answers)} more unique answers")
            print(f"\n✅ Self-consistency rate: {consistency_rate:.3f}")
            print(f"{'='*80}\n")
        
        # Save incrementally every 10 questions
        if (idx + 1) % 10 == 0:
            with open(save_path, 'w') as f:
                json.dump(self_consistency_rates, f, indent=2)
            avg_rate = sum(self_consistency_rates.values()) / len(self_consistency_rates)
            print(f"\n  💾 Saved at {idx + 1}/{len(qa_dataset)} | Avg consistency rate: {avg_rate:.3f}")
    
    torch.cuda.empty_cache()
    
    # Final save
    with open(save_path, 'w') as f:
        json.dump(self_consistency_rates, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ COMPLETED! Final save: {save_path}")
    print(f"✓ Total questions processed: {len(self_consistency_rates)}")
    avg_consistency = sum(self_consistency_rates.values()) / len(self_consistency_rates)
    print(f"✓ Average self-consistency rate: {avg_consistency:.3f}")
    print(f"{'='*60}")
    
    return self_consistency_rates

