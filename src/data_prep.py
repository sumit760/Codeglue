"""
data_prep.py
============
Data preparation for the Code Vulnerability Detection pipeline.

Loads the CodeXGLUE defect-detection dataset, normalises whitespace,
deduplicates the train split, tokenizes, and computes balanced class
weights. Everything here is importable by ``train.py``; running this file
directly prints a short EDA summary.

Configuration is read from the environment so the container can be
reconfigured without editing code. Defaults match the original notebook.
"""

import os
import re
import json
import collections

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, set_seed
from sklearn.utils.class_weight import compute_class_weight

# --------------------------------------------------------------------------- #
# Configuration (environment-overridable)
# --------------------------------------------------------------------------- #
SEED = int(os.environ.get("SEED", "42"))
PRIMARY_MODEL = os.environ.get("PRIMARY_MODEL", "huggingface/CodeBERTa-small-v1")
LARGE_MODEL = os.environ.get("LARGE_MODEL", "microsoft/codebert-base")
USE_QLORA = os.environ.get("USE_QLORA", "false").lower() == "true"
MODEL_NAME = LARGE_MODEL if USE_QLORA else PRIMARY_MODEL
DATASET_NAME = os.environ.get("DATASET_NAME", "google/code_x_glue_cc_defect_detection")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "./outputs")
DATA_DIR = os.environ.get("DATA_DIR", "./data")

# Label mapping is the single source of truth for the whole pipeline.
ID2LABEL = {0: "CLEAN", 1: "VULNERABLE"}
LABEL2ID = {"CLEAN": 0, "VULNERABLE": 1}


# --------------------------------------------------------------------------- #
# Text cleaning
# --------------------------------------------------------------------------- #
def clean_text(code: str) -> str:
    """Normalise whitespace so indentation-style differences don't add noise.

    - Collapse runs of spaces/tabs into a single space.
    - Collapse 3+ blank lines into 2.

    NOTE: inference.py mirrors this logic so train/serve preprocessing match.
    """
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def _clean_example(example):
    example["func"] = clean_text(example["func"])
    return example


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def load_clean_dataset():
    """Load the dataset, cast labels to int, clean text, and dedup the train split."""
    set_seed(SEED)
    raw = load_dataset(DATASET_NAME)
    raw = raw.map(lambda x: {"target": int(x["target"])})

    dataset = raw.map(_clean_example, desc="Cleaning whitespace")

    seen = set()

    def tag_dupes(example):
        h = hash(example["func"])
        keep = h not in seen
        seen.add(h)
        example["_keep"] = keep
        return example

    dataset["train"] = dataset["train"].map(tag_dupes, desc="Deduplication")
    before = len(dataset["train"])
    dataset["train"] = dataset["train"].filter(lambda x: x["_keep"])
    dataset["train"] = dataset["train"].remove_columns(["_keep"])
    removed = before - len(dataset["train"])
    print(f"Removed {removed} duplicates ({removed / before * 100:.1f}%). "
          f"Final train: {len(dataset['train']):,}")
    return dataset


def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def build_tokenized(dataset, tokenizer):
    """Tokenize ``func`` and rename ``target`` -> ``labels`` for the Trainer."""
    def tokenize_batch(batch):
        return tokenizer(
            batch["func"], truncation=True,
            max_length=MAX_LENGTH, padding=False,
        )

    tokenized = dataset.map(
        tokenize_batch, batched=True, remove_columns=["func"], desc="Tokenizing"
    )
    tokenized = tokenized.rename_column("target", "labels")
    tokenized.set_format("torch")
    return tokenized


def compute_weights(tokenized) -> torch.Tensor:
    """Balanced class weights for the weighted cross-entropy loss."""
    train_labels = np.array(tokenized["train"]["labels"])
    weights_np = compute_class_weight(
        class_weight="balanced", classes=np.array([0, 1]), y=train_labels
    )
    return torch.tensor(weights_np, dtype=torch.float)


def save_label_map(path: str | None = None) -> str:
    """Persist the id->label map (handy for downstream tooling / debugging)."""
    path = path or os.path.join(DATA_DIR, "id2label.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({str(k): v for k, v in ID2LABEL.items()}, f)
    return path


def prepare():
    """One-call helper used by train.py.

    Returns:
        dataset       - cleaned DatasetDict (keeps ``func`` + ``target``)
        tokenized     - tokenized DatasetDict (``input_ids`` + ``labels``)
        tokenizer     - the tokenizer
        class_weights - torch.Tensor of shape (2,)
    """
    dataset = load_clean_dataset()
    tokenizer = get_tokenizer()
    tokenized = build_tokenized(dataset, tokenizer)
    class_weights = compute_weights(tokenized)
    save_label_map()
    return dataset, tokenized, tokenizer, class_weights


# --------------------------------------------------------------------------- #
# Standalone EDA
# --------------------------------------------------------------------------- #
def _print_eda():
    set_seed(SEED)
    print("Loading dataset…")
    raw = load_dataset(DATASET_NAME)
    raw = raw.map(lambda x: {"target": int(x["target"])})

    print("\nClass distribution")
    print("-" * 50)
    for split in ["train", "validation", "test"]:
        labels = raw[split]["target"]
        cnts = collections.Counter(labels)
        total = len(labels)
        print(
            f"  {split.upper():10s}: {total:6,} total | "
            f"Clean {cnts[0]:5,} ({cnts[0] / total * 100:4.1f}%) | "
            f"Vulnerable {cnts[1]:5,} ({cnts[1] / total * 100:4.1f}%)"
        )

    tok = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
    lengths = [len(tok(ex["func"])["input_ids"]) for ex in raw["train"].select(range(500))]
    print(
        f"\nToken lengths (first 500): mean={np.mean(lengths):.0f}  "
        f"med={np.median(lengths):.0f}  p95={np.percentile(lengths, 95):.0f}  "
        f"max={max(lengths)}"
    )
    print(f"→ MAX_LENGTH={MAX_LENGTH} covers the 95th percentile cleanly.")

    # Run the full prep once so dedup stats print too.
    print()
    prepare()


if __name__ == "__main__":
    _print_eda()
