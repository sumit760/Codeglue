import collections
import json
import os

import numpy as np
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer, set_seed

from utils import (
    DATASET_NAME,
    DATA_DIR,
    ID2LABEL,
    MAX_LENGTH,
    PRIMARY_MODEL,
    clean_code,
    make_dedup_tagger,
)

set_seed(42)

print("Loading dataset...")
raw = load_dataset(DATASET_NAME)
raw = raw.map(lambda x: {"target": int(x["target"])})
print(raw)

print("\nClass distribution")
print("-" * 56)
for split in ["train", "validation", "test"]:
    labels = raw[split]["target"]
    cnts   = collections.Counter(labels)
    total  = len(labels)
    print(
        f"  {split.upper():12s}: {total:6,} total | "
        f"Clean {cnts[0]:5,} ({cnts[0]/total*100:4.1f}%) | "
        f"Vulnerable {cnts[1]:4,} ({cnts[1]/total*100:4.1f}%)"
    )

print("\nCleaning whitespace...")
dataset = raw.map(clean_code, desc="Cleaning whitespace")

print("Deduplicating train split...")
tagger           = make_dedup_tagger()
dataset["train"] = dataset["train"].map(tagger, desc="Tagging duplicates")
before           = len(dataset["train"])
dataset["train"] = dataset["train"].filter(lambda x: x["_keep"])
dataset["train"] = dataset["train"].remove_columns(["_keep"])
removed          = before - len(dataset["train"])
print(f"Removed {removed:,} duplicates ({removed/before*100:.1f}%). Final train size: {len(dataset['train']):,}")

print("\nAnalysing token lengths on first 500 train samples...")
_tok     = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
_lengths = [
    len(_tok(dataset["train"][i]["func"])["input_ids"])
    for i in range(min(500, len(dataset["train"])))
]
print(
    f"  mean={np.mean(_lengths):.0f}  "
    f"med={np.median(_lengths):.0f}  "
    f"p95={np.percentile(_lengths, 95):.0f}  "
    f"max={max(_lengths)}"
)

print("\nTokenising...")
tokenizer = AutoTokenizer.from_pretrained(PRIMARY_MODEL)

def tokenize_batch(batch):
    return tokenizer(batch["func"], truncation=True, max_length=MAX_LENGTH, padding=False)

cols_to_drop = ["func", "id", "project", "commit_id"]
tokenized    = dataset.map(
    tokenize_batch,
    batched=True,
    remove_columns=cols_to_drop,
    desc="Tokenizing",
)
tokenized = tokenized.rename_column("target", "labels")
tokenized.set_format("torch")
print(f"\nTokenized splits:\n{tokenized}")

train_labels     = np.array(tokenized["train"]["labels"])
class_weights_np = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=train_labels
).astype(np.float32)
print(f"\nClass weights -> Clean: {class_weights_np[0]:.4f}  |  Vulnerable: {class_weights_np[1]:.4f}")

os.makedirs(DATA_DIR, exist_ok=True)

dataset_path = os.path.join(DATA_DIR, "processed")
tokenized.save_to_disk(dataset_path)
print(f"\nDataset saved -> {dataset_path}")

weights_path = os.path.join(DATA_DIR, "class_weights.npy")
np.save(weights_path, class_weights_np)
print(f"Class weights saved -> {weights_path}")

labels_path = os.path.join(DATA_DIR, "id2label.json")
with open(labels_path, "w") as f:
    json.dump({str(k): v for k, v in ID2LABEL.items()}, f, indent=2)
print(f"Label map saved -> {labels_path}")

print("\nprepare_data.py complete. Run train.py next.")