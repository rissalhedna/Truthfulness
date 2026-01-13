# Discriminative Self-Verification Training

Training script for improving LLM calibration using discriminative self-verification.

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Training

```bash
python train.py
```

### Custom Configuration

```bash
python train.py \
  --model_name "meta-llama/Llama-3.1-8B-Instruct" \
  --num_epochs 1 \
  --num_self_questions 40 \
  --learning_rate 1e-5 \
  --batch_size 16 \
  --output_dir ./outputs
```

## Configuration Options

### Model Settings
- `--model_name`: HuggingFace model name (default: meta-llama/Llama-3.1-8B-Instruct)

### LoRA Settings
- `--lora_r`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha (default: 32)
- `--lora_dropout`: LoRA dropout (default: 0.05)

### Training Settings
- `--learning_rate`: Learning rate (default: 1e-5)
- `--num_epochs`: Number of training epochs (default: 1)
- `--gradient_clip`: Gradient clipping value (default: 0.5)

### Data Settings
- `--num_self_questions`: Number of self-generated questions (default: 40)
- `--n_samples_consistency`: Samples for self-consistency (default: 10)

### Evaluation Settings
- `--eval_every`: Evaluate every N steps (default: 20)
- `--max_eval_questions`: Max questions for final eval (default: 100)

### Batch Settings
- `--batch_size`: Batch size for pre-computation (default: 16)

### Output Settings
- `--output_dir`: Output directory (default: ./outputs)
- `--save_model`: Save trained model (flag)
- `--seed`: Random seed (default: 42)

## Pipeline Steps

1. **Load Data**: Load TriviaQA validation dataset
2. **Load Models**: Load base model and tokenizer
3. **Generate Self-Questions**: Generate diverse questions from model
4. **Pre-compute Labels**: 
   - Generate answers with verbalized confidence
   - Get discriminative judgments
   - Calculate self-consistency
5. **Setup Training**: Apply LoRA adapters
6. **Train**: Policy gradient training with Brier score rewards
7. **Evaluate**: Compare baseline vs trained on TriviaQA
8. **Save**: Save model and results

## Output Files

All outputs are saved to `--output_dir`:

- `config.json`: Training configuration
- `self_questions.json`: Generated self-questions
- `labeled_training_data.json`: Pre-computed labels
- `training_stats.json`: Training statistics
- `evaluation_results.json`: Final evaluation results
- `trained_model/`: Saved model (if --save_model used)

## Example Training Run

```bash
# Quick test with minimal questions
python train.py \
  --num_self_questions 20 \
  --num_epochs 1 \
  --max_eval_questions 50 \
  --output_dir ./test_outputs

# Full training
python train.py \
  --num_self_questions 250 \
  --num_epochs 3 \
  --max_eval_questions 100 \
  --batch_size 16 \
  --save_model \
  --output_dir ./full_outputs
```

## Module Structure

- `config.py`: Configuration management
- `data_utils.py`: Data loading and preprocessing
- `model_utils.py`: Model loading and LoRA setup
- `training_utils.py`: Label pre-computation and training loop
- `evaluation_utils.py`: Calibration evaluation
- `train.py`: Main training script

## Hardware Requirements

- GPU with at least 24GB VRAM (for Llama 3.1-8B)
- Adjust `--batch_size` based on available memory

