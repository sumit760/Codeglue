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

Here is the formal documentation drafted from our previous steps. You can copy and paste this directly into your final submission report.

### **1. Data Cleaning Decisions**

* **Missing Values:** The CodeXGLUE defect detection dataset is pre-curated, meaning complex imputation strategies were not required. However, non-essential metadata columns (`id`, `project`, `commit_id`) were intentionally dropped from the `DatasetDict` to optimize memory allocation and processing speed during training.
* **Normalization Strategy:** * **Text Cleaning:** Raw C/C++ code inputs contained excessive whitespace and consecutive newline characters (`\n\n`). This was normalized by replacing multiple spaces and newlines with a single space to prevent the tokenizer from wasting context space on empty formatting blocks.
* **Tokenization & Truncation:** The cleaned code was tokenized using `CodeBERTa-small-v1`'s native tokenizer. All sequences were padded and truncated to a `max_length` of 512 tokens to ensure uniform matrix dimensions while staying strictly within the model's maximum context window.
* **Label Transformation:** The target feature was converted from boolean values (`True`/`False`) into standard integer labels (`1`/`0`) required by the PyTorch training loop for binary classification.


* **Class Distribution:** The close alignment between the F1 score and the raw Accuracy metric during our evaluation indicates that the validation/test splits are relatively balanced. This ensures the model learned to recognize syntax defects rather than simply defaulting to predicting a majority class.

---

### **2. Model Selection Rationale**

**Selected Model:** `huggingface/CodeBERTa-small-v1`

* **Domain Specificity:** According to the official Hugging Face model card, CodeBERTa is a RoBERTa-like model specifically pre-trained on the CodeSearchNet dataset. This endows the model with a native, structural understanding of programming languages (including C/C++), making it vastly superior for code analysis compared to general-purpose English NLP models.
* **Resource Efficiency:** The assignment specifications require keeping the model footprint under 200 MB. CodeBERTa-small-v1 fulfills this perfectly (approximately 84 million parameters). Its compact architecture allowed for smooth, highly efficient training on Kaggle's free T4 GPUs utilizing mixed precision (`fp16=True`), completely avoiding Out-Of-Memory (OOM) bottlenecks.

---

### **3. Experiment Comparison**

| Metric | Version 1 (LR: 3e-5) | Version 2 (LR: 5e-5) |
| --- | --- | --- |
| **Validation Loss** | 1.284 | 1.504 |
| **Accuracy** | 62.88% | 62.26% |
| **F1 Score** | 62.93% | 62.24% |

**Performance Analysis:**
Version 1 outperformed Version 2 across all logged metrics. By increasing the learning rate from 3e-5 to 5e-5 in the second run, the validation loss increased significantly (from 1.284 to 1.504), accompanied by a slight decay in both Accuracy and F1 scores. This indicates that the more aggressive learning rate in Version 2 was too large, causing the optimizer to step past the optimal model weights (overshooting the local minima in the loss landscape). Version 1's smaller step size allowed for more stable and accurate convergence.

---

I would love to test your understanding of the hyperparameter optimization we just analyzed with a quick quiz, but you previously asked me to use a specific format for quizzes without providing the actual format details. Could you clarify how you would like those questions structured?
