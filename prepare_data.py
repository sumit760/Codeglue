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
    MODEL_NAME,
    PRIMARY_MODEL,
    clean_code,
    make_dedup_tagger,
)

set_seed(42)

print("Loading dataset...")
raw = load_dataset(DATASET_NAME)
raw = raw.map(lambda x: {"target": int(x["target"])})
print(raw)
print(f"Columns: {raw['train'].column_names}")
print(f"\nSample (first 300 chars):\n{raw['train'][0]['func'][:300]}")
print(f"Label: {raw['train'][0]['target']}  ->  0 = clean, 1 = vulnerable")

print("Class distribution")
print("-" * 50)
for split in ["train", "validation", "test"]:
    labels = raw[split]["target"]
    cnts   = collections.Counter(labels)
    total  = len(labels)
    print(
        f"  {split.upper():8s}: {total:6,} total | "
        f"Clean {cnts[0]:5,} ({cnts[0]/total*100:4.1f}%) | "
        f"Vulnerable {cnts[1]:4,} ({cnts[1]/total*100:4.1f}%)"
    )

_tok     = AutoTokenizer.from_pretrained(PRIMARY_MODEL)
_lengths = [len(_tok(ex["func"])["input_ids"]) for ex in raw["train"].select(range(500))]
print(
    f"Token lengths (first 500): "
    f"mean={np.mean(_lengths):.0f}  "
    f"med={np.median(_lengths):.0f}  "
    f"p95={np.percentile(_lengths, 95):.0f}  "
    f"max={max(_lengths)}"
)
print(f"MAX_LENGTH={MAX_LENGTH}")

dataset = raw.map(clean_code, desc="Cleaning whitespace")

seen = set()
tagger           = make_dedup_tagger()
dataset["train"] = dataset["train"].map(tagger, desc="Deduplication")
before           = len(dataset["train"])
dataset["train"] = dataset["train"].filter(lambda x: x["_keep"])
dataset["train"] = dataset["train"].remove_columns(["_keep"])
removed          = before - len(dataset["train"])
print(f"Removed {removed} duplicates ({removed/before*100:.1f}%). Final train: {len(dataset['train']):,}")

os.makedirs(DATA_DIR, exist_ok=True)
with open(f"{DATA_DIR}/id2label.json", "w") as f:
    json.dump({str(k): v for k, v in ID2LABEL.items()}, f)
print(f"Saved {DATA_DIR}/id2label.json:", ID2LABEL)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"Tokenizer: {tokenizer.__class__.__name__}  |  Vocab: {tokenizer.vocab_size:,}")

def tokenize_batch(batch):
    return tokenizer(batch["func"], truncation=True, max_length=MAX_LENGTH, padding=False)

tokenized = dataset.map(tokenize_batch, batched=True, remove_columns=["func"], desc="Tokenizing")
tokenized = tokenized.rename_column("target", "labels")
tokenized.set_format("torch")
print(f"\nTokenized:\n{tokenized}")

train_labels     = np.array(tokenized["train"]["labels"])
class_weights_np = compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=train_labels
).astype(np.float32)
print(f"Class weights — Clean: {class_weights_np[0]:.3f}  |  Vulnerable: {class_weights_np[1]:.3f}")

tokenized.save_to_disk(f"{DATA_DIR}/processed")
print(f"Dataset saved -> {DATA_DIR}/processed")

np.save(f"{DATA_DIR}/class_weights.npy", class_weights_np)
print(f"Class weights saved -> {DATA_DIR}/class_weights.npy")

dataset.save_to_disk(f"{DATA_DIR}/cleaned")
print(f"Cleaned dataset saved -> {DATA_DIR}/cleaned")

print("\nprepare_data.py complete. Run train.py next.")
