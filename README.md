# CodeBERTa Vulnerability Detector

A fine-tuned CodeBERTa-small-v1 model for classifying C/C++ code snippets as CLEAN or VULNERABLE, trained on the CodeXGLUE defect detection dataset.

## Files

### `prepare_data.py`
Downloads and preprocesses the CodeXGLUE defect detection dataset. Cleans whitespace, deduplicates training samples, tokenizes the code with the CodeBERTa tokenizer, computes class weights for handling imbalanced data, and saves the processed dataset to `./data/`.

### `train.py`
Trains two experimental configurations of the CodeBERTa model:
- **V1**: Linear learning rate schedule, 3 epochs, LR=2e-5, batch size=16
- **V2**: Cosine schedule with warmup, 5 epochs, LR=5e-5, batch size=32
Evaluates both on the test set, selects the best model by weighted F1 score, and pushes it to the Hugging Face Hub. Includes an optional QLoRA path for fine-tuning the larger CodeBERT-base model with INT4 quantization.

### `inference.py`
Loads the fine-tuned model from the Hugging Face Hub and classifies C/C++ code snippets. Supports three modes: interactive demo with built-in snippets, single snippet classification via `--snippet`, and file classification via `--file`.

### `utils.py`
Shared utilities including:
- Configuration constants (model names, paths, label mappings)
- `load_secrets()` for authenticating with Hugging Face and WandB
- `clean_code()` for normalizing whitespace in source code
- `make_dedup_tagger()` for identifying duplicate samples
- `compute_metrics()` for evaluating accuracy, precision, recall, and F1 scores
- `WeightedTrainer` - a custom Trainer class that applies class-weighted loss

## Usage

```bash
# Step 1: Prepare the dataset
python prepare_data.py

# Step 2: Train the model
python train.py

# Step 3: Run inference
python inference.py                    # Demo mode
python inference.py --snippet "gets(buf);"    # Single snippet
python inference.py --file sample.c          # From file
```
