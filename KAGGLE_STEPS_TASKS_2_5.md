# Kaggle Execution Steps For Tasks 2, 3, 4, and 5

These steps cover only the assigned group tasks:

- Task 2: data preparation and normalisation
- Task 3: model selection and loading
- Task 4: two Kaggle experiment versions tracked with W&B
- Task 5: push the best model to Hugging Face Hub

## Assumed Dataset And Model

Dataset: `google/code_x_glue_cc_defect_detection`

Model: `huggingface/CodeBERTa-small-v1`

Reason: the dataset is a compact C-code defect detection dataset, and CodeBERTa-small is trained on source code, so it is better matched to this task than a natural-language model.

Sources:

- Dataset card: https://huggingface.co/datasets/google/code_x_glue_cc_defect_detection
- Model card: https://huggingface.co/huggingface/CodeBERTa-small-v1

## 1. Create Kaggle Notebook

1. Open Kaggle.
2. Create a new notebook.
3. In notebook settings, set `Accelerator` to `GPU T4`.
4. Turn Internet on.
5. Upload or clone the repository files.

If using GitHub:

```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME/group_assignment_tasks_2_5
```

## 2. Add Kaggle Secrets

In Kaggle notebook:

1. Open `Add-ons`.
2. Open `Secrets`.
3. Add `WANDB_API_KEY`.
4. Add `HF_TOKEN`.
5. Enable both secrets for the notebook.

Do not paste tokens directly into notebook code.

## 3. Install Dependencies

Run this cell:

```python
!pip -q install -r requirements.txt
```

## 4. Prepare Data

Run:

```python
!python src/prepare_data.py \
  --dataset-name google/code_x_glue_cc_defect_detection \
  --dataset-config default \
  --output-dir data/processed/code_defect \
  --artifact-dir artifacts
```

Files produced:

- `artifacts/id2label.json`
- `artifacts/label2id.json`
- `artifacts/raw_data_profile.json`
- `artifacts/cleaning_report.json`
- `data/processed/code_defect/`

Commit `artifacts/id2label.json` to GitHub. Do not commit `data/processed/code_defect/`.

## 5. Run Experiment Version 1

Run:

```python
!python src/train_kaggle.py \
  --dataset-dir data/processed/code_defect \
  --id2label-path artifacts/id2label.json \
  --model-name huggingface/CodeBERTa-small-v1 \
  --project mlops-group-project \
  --run-name codeberta-defect-v1 \
  --version v1 \
  --output-dir outputs/run-v1 \
  --epochs 2 \
  --learning-rate 3e-5 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256
```

## 6. Run Experiment Version 2

This version changes the learning rate and number of epochs.

```python
!python src/train_kaggle.py \
  --dataset-dir data/processed/code_defect \
  --id2label-path artifacts/id2label.json \
  --model-name huggingface/CodeBERTa-small-v1 \
  --project mlops-group-project \
  --run-name codeberta-defect-v2 \
  --version v2 \
  --output-dir outputs/run-v2 \
  --epochs 3 \
  --learning-rate 2e-5 \
  --train-batch-size 8 \
  --eval-batch-size 16 \
  --max-length 256
```

## 7. Compare Metrics

Run:

```python
!cat outputs/run-v1/metrics.json
!cat outputs/run-v2/metrics.json
```

Choose the better version using `test_f1` first. If F1 is close, use `test_accuracy` and `test_loss` as secondary checks.

## 8. Push Best Model To Hugging Face

If Version 1 is better:

```python
!python src/push_to_hub.py \
  --model-dir outputs/run-v1/best_model \
  --hf-repo YOUR_HF_USERNAME/codeberta-defect-detection-mlops \
  --project mlops-group-project \
  --run-name deploy-codeberta-defect
```

If Version 2 is better:

```python
!python src/push_to_hub.py \
  --model-dir outputs/run-v2/best_model \
  --hf-repo YOUR_HF_USERNAME/codeberta-defect-detection-mlops \
  --project mlops-group-project \
  --run-name deploy-codeberta-defect
```

## 9. Capture Evidence For Report

Save these outputs:

- Screenshot of Kaggle notebook showing successful Version 1 run.
- Screenshot of Kaggle notebook showing successful Version 2 run.
- Screenshot of W&B project dashboard showing both runs.
- Public W&B project link.
- Public Hugging Face model link.
- `artifacts/raw_data_profile.json` values for data inspection.
- `artifacts/cleaning_report.json` values for cleaning decisions.
- `outputs/run-v1/metrics.json`.
- `outputs/run-v2/metrics.json`.

## 10. Make Links Public

Check all links in a private/incognito browser:

- Kaggle Notebook Version 1 link
- Kaggle Notebook Version 2 link
- W&B dashboard link
- Hugging Face model link
- GitHub repository link
