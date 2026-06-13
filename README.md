# CodeBERTa Vulnerability Detector

Classify C/C++ code snippets as CLEAN or VULNERABLE using a fine-tuned CodeBERTa-small-v1 model on the CodeXGLUE defect detection dataset.

---

## Project Structure

| # | File | Description |
|---|------|-------------|
| 1 | data_prep.py | Downloads and preprocesses the CodeXGLUE dataset -- cleans whitespace, deduplicates samples, tokenizes with CodeBERTa, computes class weights |
| 2 | train.py | Trains two experimental configurations (V1 and V2), picks the best by weighted F1, pushes to Hugging Face Hub. Optional QLoRA path included |
| 3 | inference.py | Loads the fine-tuned model and classifies code snippets via demo, CLI --snippet, or --file modes |

---

## Experiments

| Config | Schedule | Epochs | LR | Batch Size |
|--------|----------|--------|----|------------|
| V1 | Linear | 3 | 2e-5 | 16 |
| V2 | Cosine + Warmup | 5 | 5e-5 | 32 |

Evaluates both on the test set and selects the best model by weighted F1 score.

---

## Quick Start

### Step 1: Prepare the dataset
```bash
python data_prep.py
```

### Step 2: Train the model
```bash
python train.py
```

### Step 3: Run inference
```bash
python inference.py                       # Interactive demo mode
python inference.py --snippet "gets(buf);"  # Single code snippet
python inference.py --file sample.c         # From a source file
```

---

## Requirements

- Python 3.12+
- PyTorch
- Transformers (Hugging Face)
- Datasets (Hugging Face)
- scikit-learn
- wandb (Weights and Biases)

---

## Model on Hugging Face Hub

The best fine-tuned model is automatically pushed to:
https://huggingface.co/pranshur10/codeberta-vuln-detector
https://huggingface.co/sumitp76/codeberta-vuln-detector
