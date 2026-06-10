import os
import json
import numpy as np
import evaluate
from datasets import load_from_disk
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding, AutoTokenizer
import wandb
from kaggle_secrets import UserSecretsClient

def train():
    try:
        secrets = UserSecretsClient()
        os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
        wandb.login()
    except:
        pass

    tokenized_dataset = load_from_disk("/kaggle/working/processed_dataset")
    model_name = "huggingface/CodeBERTa-small-v1"
    
    with open("/kaggle/working/id2label.json", "r") as f:
        id2label = {int(k): v for k, v in json.load(f).items()}
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, id2label=id2label, label2id={v: k for k, v in id2label.items()})
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    accuracy_metric, f1_metric = evaluate.load("accuracy"), evaluate.load("f1")
    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred[0], axis=1)
        return {
            "eval_accuracy": accuracy_metric.compute(predictions=preds, references=eval_pred[1])["accuracy"],
            "eval_f1": f1_metric.compute(predictions=preds, references=eval_pred[1], average="binary")["f1"]
        }

    training_args = TrainingArguments(
        output_dir="/kaggle/working/results", learning_rate=3e-5, per_device_train_batch_size=16,
        num_train_epochs=3, eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_f1", fp16=True, report_to="wandb"
    )

    trainer = Trainer(
        model=model, args=training_args, train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"], processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer), compute_metrics=compute_metrics,
    )
    trainer.train()
    trainer.save_model("/kaggle/working/best_model")

if __name__ == "__main__":
    train()
