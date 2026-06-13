import os
import re
import numpy as np
import torch
from transformers import Trainer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
)

PRIMARY_MODEL = "huggingface/CodeBERTa-small-v1"
LARGE_MODEL   = "microsoft/codebert-base"
DATASET_NAME  = "google/code_x_glue_cc_defect_detection"
HF_REPO_ID    = "pranshur10/codeberta-vuln-detector"
WANDB_PROJECT = "mlops-a3-vuln-detection"
MAX_LENGTH    = 512
OUTPUT_DIR    = "./outputs"
DATA_DIR      = "./data"

ID2LABEL = {0: "CLEAN", 1: "VULNERABLE"}
LABEL2ID = {"CLEAN": 0, "VULNERABLE": 1}


def load_secrets():
    try:
        from kaggle_secrets import UserSecretsClient
        _s = UserSecretsClient()
        os.environ["HF_TOKEN"]      = _s.get_secret("HF_TOKEN")
        os.environ["WANDB_API_KEY"] = _s.get_secret("WANDB_API_KEY")
        print("Kaggle secrets loaded.")
    except Exception:
        assert os.environ.get("WANDB_API_KEY"), "WANDB_API_KEY not set."
        assert os.environ.get("HF_TOKEN"),      "HF_TOKEN not set."


def clean_code(example):
    code = example["func"]
    code = re.sub(r"[ \t]+",  " ",   code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    example["func"] = code.strip()
    return example


def make_dedup_tagger():
    seen = set()

    def tag_dupes(example):
        h = hash(example["func"])
        example["_keep"] = h not in seen
        seen.add(h)
        return example

    return tag_dupes


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
        "f1_vuln":     round(f1_score(labels, preds, pos_label=1, average="binary", zero_division=0), 4),
        "precision":   round(precision_score(labels, preds, average="weighted", zero_division=0), 4),
        "recall":      round(recall_score(labels, preds, average="weighted"), 4),
    }


class WeightedTrainer(Trainer):

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels  = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits  = outputs.logits
        weights = self._class_weights.to(logits.device)
        loss    = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss