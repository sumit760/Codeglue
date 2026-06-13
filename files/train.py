import os

import numpy as np
import torch
import wandb
from datasets import load_dataset, load_from_disk
from huggingface_hub import login
from sklearn.metrics import classification_report
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    TrainingArguments,
    set_seed,
)
from transformers.integrations import WandbCallback

from utils import (
    DATA_DIR,
    DATASET_NAME,
    HF_REPO_ID,
    ID2LABEL,
    LABEL2ID,
    LARGE_MODEL,
    MAX_LENGTH,
    MODEL_NAME,
    OUTPUT_DIR,
    USE_QLORA,
    WANDB_PROJECT,
    WeightedTrainer,
    clean_code,
    compute_metrics,
    load_secrets,
    make_dedup_tagger,
)

set_seed(42)

load_secrets()
wandb.login(key=os.environ["WANDB_API_KEY"], relogin=True)
login(token=os.environ["HF_TOKEN"])
print("Authenticated with W&B and Hugging Face.")

print("\nLoading prepared data from disk...")
tokenized     = load_from_disk(f"{DATA_DIR}/processed")
class_weights = torch.tensor(
    np.load(f"{DATA_DIR}/class_weights.npy"),
    dtype=torch.float,
)
print(f"Class weights — Clean: {class_weights[0]:.3f}  |  Vulnerable: {class_weights[1]:.3f}")

tokenizer     = AutoTokenizer.from_pretrained(MODEL_NAME)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

v1_config = {
    "model":         MODEL_NAME,   "dataset": DATASET_NAME,  "version":  "v1",
    "learning_rate": 2e-5,         "epochs":  3,             "batch_size": 16,
    "warmup_steps":  0,            "scheduler": "linear",    "weight_decay": 0.01,
    "max_length":    MAX_LENGTH,
}

wandb.init(project=WANDB_PROJECT, name="run-v1", config=v1_config, reinit=True)

args_v1 = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/v1",        run_name="run-v1",
    num_train_epochs=v1_config["epochs"],
    per_device_train_batch_size=v1_config["batch_size"],
    per_device_eval_batch_size=32,
    learning_rate=v1_config["learning_rate"],
    weight_decay=v1_config["weight_decay"],
    warmup_steps=v1_config["warmup_steps"],
    lr_scheduler_type=v1_config["scheduler"],
    eval_strategy="epoch",   save_strategy="epoch",
    logging_steps=50,        load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",  greater_is_better=True,
    report_to="wandb",       fp16=torch.cuda.is_available(),
    seed=42,                 dataloader_num_workers=2,
)

model_v1 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

trainer_v1 = WeightedTrainer(
    class_weights=class_weights,  model=model_v1,  args=args_v1,
    train_dataset=tokenized["train"],  eval_dataset=tokenized["validation"],
    processing_class=tokenizer,   data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer_v1.train()
wandb.finish()

v2_config = {
    "model":         MODEL_NAME,   "dataset": DATASET_NAME,  "version":  "v2",
    "learning_rate": 5e-5,         "epochs":  5,             "batch_size": 32,
    "warmup_steps":  200,          "scheduler": "cosine",    "weight_decay": 0.05,
    "max_length":    MAX_LENGTH,
}

wandb.init(project=WANDB_PROJECT, name="run-v2", config=v2_config, reinit=True)

args_v2 = TrainingArguments(
    output_dir=f"{OUTPUT_DIR}/v2",        run_name="run-v2",
    num_train_epochs=v2_config["epochs"],
    per_device_train_batch_size=v2_config["batch_size"],
    per_device_eval_batch_size=64,
    learning_rate=v2_config["learning_rate"],
    weight_decay=v2_config["weight_decay"],
    warmup_steps=v2_config["warmup_steps"],
    lr_scheduler_type=v2_config["scheduler"],
    eval_strategy="epoch",   save_strategy="epoch",
    logging_steps=50,        load_best_model_at_end=True,
    metric_for_best_model="f1_weighted",  greater_is_better=True,
    report_to="wandb",       fp16=torch.cuda.is_available(),
    seed=42,                 dataloader_num_workers=2,
)

model_v2 = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
    ignore_mismatched_sizes=True,
)

trainer_v2 = WeightedTrainer(
    class_weights=class_weights,  model=model_v2,  args=args_v2,
    train_dataset=tokenized["train"],  eval_dataset=tokenized["validation"],
    processing_class=tokenizer,   data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer_v2.train()
wandb.finish()

trainer_v1.remove_callback(WandbCallback)
trainer_v2.remove_callback(WandbCallback)

trainers = {"V1": trainer_v1, "V2": trainer_v2}
results  = {}

for name, trainer in trainers.items():
    r = trainer.evaluate(eval_dataset=tokenized["test"])
    results[name] = r
    print(f"\n{name}")
    print(f"  Accuracy    : {r['eval_accuracy']:.4f}")
    print(f"  F1 Weighted : {r['eval_f1_weighted']:.4f}")
    print(f"  F1 Vuln     : {r['eval_f1_vuln']:.4f}")
    print(f"  Precision   : {r['eval_precision']:.4f}")
    print(f"  Recall      : {r['eval_recall']:.4f}")

best_name    = max(results, key=lambda k: results[k]["eval_f1_weighted"])
best_trainer = trainers[best_name]
print(f"\nBest model: {best_name}  (weighted F1 = {results[best_name]['eval_f1_weighted']:.4f})")

preds_out = best_trainer.predict(tokenized["test"])
y_true    = preds_out.label_ids
y_pred    = np.argmax(preds_out.predictions, axis=-1)
print(f"Classification report - {best_name}\n")
print(classification_report(y_true, y_pred, target_names=["CLEAN", "VULNERABLE"], digits=4))

print(f"Pushing {best_name} -> {HF_REPO_ID}")
best_trainer.push_to_hub(
    repo_id=HF_REPO_ID,
    commit_message=f"CodeBERTa fine-tuned on CodeXGLUE defect detection ({best_name})",
    blocking=True,
)
tokenizer.push_to_hub(HF_REPO_ID)

model_url    = f"https://huggingface.co/{HF_REPO_ID}"
_summary_run = wandb.init(project=WANDB_PROJECT, name="hub-push-summary", reinit=True)
_summary_run.summary["huggingface_model"] = model_url
wandb.finish()
print(f"Model live at: {model_url}")

if not USE_QLORA:
    print("USE_QLORA is False - skipping QLoRA section.")
else:
    from transformers import BitsAndBytesConfig
    from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,  bnb_4bit_use_double_quant=True,
    )
    model_qlora = AutoModelForSequenceClassification.from_pretrained(
        LARGE_MODEL, quantization_config=bnb_config, device_map="auto",
        num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID,
        ignore_mismatched_sizes=True,
    )
    model_qlora = prepare_model_for_kbit_training(model_qlora)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  r=8,  lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.1,  bias="none",
    )
    model_qlora = get_peft_model(model_qlora, lora_config)
    model_qlora.print_trainable_parameters()

    tokenizer_qlora = AutoTokenizer.from_pretrained(LARGE_MODEL)

    dataset = load_from_disk(f"{DATA_DIR}/cleaned")

    def tokenize_qlora(batch):
        return tokenizer_qlora(batch["func"], truncation=True, max_length=MAX_LENGTH, padding=False)

    tokenized_qlora = dataset.map(
        tokenize_qlora, batched=True, remove_columns=["func"], desc="Tokenizing QLoRA"
    )
    tokenized_qlora = tokenized_qlora.rename_column("target", "labels")
    tokenized_qlora.set_format("torch")

    wandb.init(project=WANDB_PROJECT, name="run-qlora-v1", reinit=True, config={
        "model": LARGE_MODEL,  "method": "QLoRA-INT4",  "lora_r": 8,
        "learning_rate": 2e-4,  "epochs": 3,  "batch_size": 16,
    })

    args_qlora = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/qlora",  run_name="run-qlora-v1",
        num_train_epochs=3,    per_device_train_batch_size=16,
        per_device_eval_batch_size=32,  learning_rate=2e-4,
        weight_decay=0.01,     warmup_steps=100,  lr_scheduler_type="cosine",
        eval_strategy="epoch", save_strategy="epoch",  logging_steps=50,
        load_best_model_at_end=True,  metric_for_best_model="f1_weighted",
        report_to="wandb",     fp16=True,  optim="paged_adamw_8bit",  seed=42,
    )

    trainer_qlora = WeightedTrainer(
        class_weights=class_weights,  model=model_qlora,  args=args_qlora,
        train_dataset=tokenized_qlora["train"],
        eval_dataset=tokenized_qlora["validation"],
        processing_class=tokenizer_qlora,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer_qlora),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )
    trainer_qlora.train()
    wandb.finish()
