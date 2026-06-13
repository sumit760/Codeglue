"""
train.py
========
Fine-tunes CodeBERTa on CodeXGLUE defect detection.

Runs two experiments (V1 conservative baseline, V2 cosine + warmup),
evaluates both on the test split, selects the higher weighted-F1 model,
saves it locally, and pushes the best model **and** tokenizer to a single
Hugging Face Hub repo.

Environment variables
---------------------
HF_TOKEN        required to push to the Hub (push is skipped if absent)
HF_REPO_ID      target repo            (default: sumitp76/codeberta-vuln-detector)
WANDB_API_KEY   optional; enables W&B logging if set
WANDB_PROJECT   W&B project name       (default: mlops-a3-vuln-detection)
OUTPUT_DIR      checkpoints / outputs  (default: ./outputs)
plus everything configurable in data_prep.py (MODEL_NAME, MAX_LENGTH, …)
"""

import os

import numpy as np
import torch
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    EarlyStoppingCallback, set_seed,
)
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, classification_report,
)
from huggingface_hub import login

import data_prep
from data_prep import (
    MODEL_NAME, DATASET_NAME, MAX_LENGTH, OUTPUT_DIR, ID2LABEL, LABEL2ID, SEED,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
HF_REPO_ID = os.environ.get("HF_REPO_ID", "sumitp76/codeberta-vuln-detector")
HF_TOKEN = os.environ.get("HF_TOKEN") or None
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "mlops-a3-vuln-detection")
WANDB_ENABLED = bool(os.environ.get("WANDB_API_KEY"))
REPORT_TO = "wandb" if WANDB_ENABLED else "none"

V1_CONFIG = {
    "version": "v1", "learning_rate": 2e-5, "epochs": 3, "batch_size": 16,
    "eval_batch_size": 32, "warmup_steps": 0, "scheduler": "linear", "weight_decay": 0.01,
}
V2_CONFIG = {
    "version": "v2", "learning_rate": 5e-5, "epochs": 5, "batch_size": 32,
    "eval_batch_size": 64, "warmup_steps": 200, "scheduler": "cosine", "weight_decay": 0.05,
}


# --------------------------------------------------------------------------- #
# Training infrastructure
# --------------------------------------------------------------------------- #
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy":    round(accuracy_score(labels, preds), 4),
        "f1_weighted": round(f1_score(labels, preds, average="weighted"), 4),
        "f1_vuln":     round(f1_score(labels, preds, pos_label=1,
                                      average="binary", zero_division=0), 4),
        "precision":   round(precision_score(labels, preds,
                                             average="weighted", zero_division=0), 4),
        "recall":      round(recall_score(labels, preds, average="weighted"), 4),
    }


class WeightedTrainer(Trainer):
    """Trainer with a class-weighted cross-entropy loss."""

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").long()
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self._class_weights.to(logits.device)
        loss = torch.nn.CrossEntropyLoss(weight=weights)(logits, labels)
        return (loss, outputs) if return_outputs else loss


def build_trainer(cfg, tokenized, tokenizer, class_weights, data_collator):
    args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{cfg['version']}",
        run_name=f"run-{cfg['version']}",
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["eval_batch_size"],
        learning_rate=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_steps"],
        lr_scheduler_type=cfg["scheduler"],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        report_to=REPORT_TO,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        dataloader_num_workers=2,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )

    return WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )


def run_experiment(cfg, tokenized, tokenizer, class_weights, data_collator):
    print(f"\n{'=' * 60}\nExperiment {cfg['version'].upper()}\n{'=' * 60}")
    run = None
    if WANDB_ENABLED:
        import wandb
        run = wandb.init(
            project=WANDB_PROJECT, name=f"run-{cfg['version']}",
            config={**cfg, "model": MODEL_NAME, "dataset": DATASET_NAME,
                    "max_length": MAX_LENGTH},
            reinit=True,
        )
    trainer = build_trainer(cfg, tokenized, tokenizer, class_weights, data_collator)
    trainer.train()
    if run is not None:
        run.finish()
    return trainer


# --------------------------------------------------------------------------- #
# Hub push (the fix: push model + tokenizer EXPLICITLY to HF_REPO_ID)
# --------------------------------------------------------------------------- #
def push_best(best_name, best_trainer, tokenizer):
    """Persist the best model locally and push it (with the tokenizer) to the Hub.

    Trainer.push_to_hub() would derive the repo from output_dir's basename and
    ignore HF_REPO_ID, so we push the model + tokenizer directly to guarantee
    config.json + weights + tokenizer all land in the same, loadable repo.
    """
    best_model = best_trainer.model
    best_model.config.id2label = ID2LABEL
    best_model.config.label2id = LABEL2ID

    local_dir = f"{OUTPUT_DIR}/best"
    best_model.save_pretrained(local_dir)
    tokenizer.save_pretrained(local_dir)
    print(f"Saved best model locally -> {local_dir}")

    if not HF_TOKEN:
        print("HF_TOKEN not set — skipping Hub push. Model available locally only.")
        return None

    login(token=HF_TOKEN)
    msg = f"CodeBERTa fine-tuned on CodeXGLUE defect detection ({best_name})"
    print(f"Pushing best model ({best_name}) -> {HF_REPO_ID}")
    best_model.push_to_hub(HF_REPO_ID, commit_message=msg, token=HF_TOKEN)
    tokenizer.push_to_hub(HF_REPO_ID, token=HF_TOKEN)
    url = f"https://huggingface.co/{HF_REPO_ID}"
    print(f"Model + tokenizer live at: {url}")
    return url


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    set_seed(SEED)
    print("PyTorch:", torch.__version__, "| CUDA:", torch.cuda.is_available())

    _, tokenized, tokenizer, class_weights = data_prep.prepare()
    print(f"Class weights — Clean: {class_weights[0]:.3f} | "
          f"Vulnerable: {class_weights[1]:.3f}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainers = {
        "V1": run_experiment(V1_CONFIG, tokenized, tokenizer, class_weights, data_collator),
        "V2": run_experiment(V2_CONFIG, tokenized, tokenizer, class_weights, data_collator),
    }

    # --- Evaluation & comparison on the held-out test split ---
    print(f"\n{'=' * 60}\nEvaluation on test split\n{'=' * 60}")
    results = {}
    for name, trainer in trainers.items():
        r = trainer.evaluate(eval_dataset=tokenized["test"])
        results[name] = r
        print(f"\n{name}")
        print(f"  Accuracy    : {r['eval_accuracy']:.4f}")
        print(f"  F1 Weighted : {r['eval_f1_weighted']:.4f}")
        print(f"  F1 Vuln     : {r['eval_f1_vuln']:.4f}")
        print(f"  Precision   : {r['eval_precision']:.4f}")
        print(f"  Recall      : {r['eval_recall']:.4f}")

    best_name = max(results, key=lambda k: results[k]["eval_f1_weighted"])
    best_trainer = trainers[best_name]
    print(f"\nBest model: {best_name} "
          f"(weighted F1 = {results[best_name]['eval_f1_weighted']:.4f})")

    preds_out = best_trainer.predict(tokenized["test"])
    y_true = preds_out.label_ids
    y_pred = np.argmax(preds_out.predictions, axis=-1)
    print(f"\nClassification report — {best_name}\n")
    print(classification_report(y_true, y_pred,
                                target_names=["CLEAN", "VULNERABLE"], digits=4))

    push_best(best_name, best_trainer, tokenizer)


if __name__ == "__main__":
    main()

