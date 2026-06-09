# Report Snippets For Your Assigned Tasks

Replace the placeholders after you run the Kaggle notebook.

## Task 2: Data Preparation And Normalisation

For this project, I used the `google/code_x_glue_cc_defect_detection` dataset from Hugging Face. It contains C source-code functions labelled for defect detection. The dataset card explains that the task is binary classification: label `1` represents insecure code and label `0` represents secure code. The dataset has three splits: 21,854 training examples, 2,732 validation examples, and 2,732 test examples.

I inspected the raw data fields, including `func`, `target`, `project`, and `commit_id`. For cleaning, I did not lowercase the code and did not remove punctuation because capitalization and symbols are meaningful in source code. I normalised line endings, removed trailing whitespace, removed empty code rows, and dropped duplicated code-label pairs. I converted the boolean `target` field into numeric labels and saved the mapping as `artifacts/id2label.json`, where `0` is `secure` and `1` is `insecure`. The prepared dataset was saved locally for Kaggle training, but the processed dataset folder was not committed to GitHub.

Add your actual cleaning numbers:

| Split | Rows Before | Empty Code Removed | Duplicate Removed | Rows After |
|-------|-------------|--------------------|-------------------|------------|
| Train | `<ADD>` | `<ADD>` | `<ADD>` | `<ADD>` |
| Validation | `<ADD>` | `<ADD>` | `<ADD>` | `<ADD>` |
| Test | `<ADD>` | `<ADD>` | `<ADD>` | `<ADD>` |

## Task 3: Model Selection Rationale

I selected `huggingface/CodeBERTa-small-v1` for this task because the input data is source code, not normal English text. The model card describes CodeBERTa as a RoBERTa-like model trained on the CodeSearchNet dataset from GitHub. It also states that the tokenizer was trained on code and can encode code more efficiently than a tokenizer trained mainly on natural language. The small version has 6 layers and 84M parameters, so it is suitable for Kaggle's free T4 GPU. Since the task is to classify C functions as secure or insecure, a compact code-pretrained model is a better match than a general text model.

Model card: https://huggingface.co/huggingface/CodeBERTa-small-v1

## Task 4: Experiment Comparison

I trained two versions in Kaggle using Hugging Face Trainer and logged both runs to W&B.

| Version | Epochs | Learning Rate | Batch Size | Max Length | Test Accuracy | Test F1 | Test Loss |
|---------|--------|---------------|------------|------------|---------------|---------|-----------|
| V1 | 2 | 3e-5 | 8 | 256 | `<ADD>` | `<ADD>` | `<ADD>` |
| V2 | 3 | 2e-5 | 8 | 256 | `<ADD>` | `<ADD>` | `<ADD>` |

Version `<ADD_BEST_VERSION>` performed better because `<ADD_REASON_FROM_METRICS>`. I selected the best model mainly using weighted F1 because the secure and insecure classes may not be perfectly balanced. Accuracy was also checked, but F1 gives a better view of how well the model handles both defect labels.

## Task 5: Hugging Face Deployment

After comparing the two versions, I pushed the best model and tokenizer to a public Hugging Face repository. The model URL was also logged to the W&B summary so the training and deployment records are connected.

- Hugging Face model: `<ADD_HF_MODEL_LINK>`
- W&B project dashboard: `<ADD_WANDB_PROJECT_LINK>`
- Kaggle Version 1 notebook: `<ADD_KAGGLE_V1_LINK>`
- Kaggle Version 2 notebook: `<ADD_KAGGLE_V2_LINK>`
