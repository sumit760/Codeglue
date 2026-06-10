import os
import json
from datasets import load_dataset
from transformers import AutoTokenizer

def clean_and_tokenize():
    dataset = load_dataset("google/code_x_glue_cc_defect_detection")
    model_name = "huggingface/CodeBERTa-small-v1"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def preprocess_function(examples):
        cleaned_code = [" ".join(code.split()) for code in examples["func"]]
        model_inputs = tokenizer(cleaned_code, truncation=True, padding="max_length", max_length=512)
        model_inputs["labels"] = [1 if label else 0 for label in examples["target"]]
        return model_inputs

    columns_to_remove = ['id', 'func', 'target', 'project', 'commit_id']
    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=columns_to_remove)
    
    output_dir = "/kaggle/working/processed_dataset"
    tokenized_dataset.save_to_disk(output_dir)
    
    with open("/kaggle/working/id2label.json", "w") as f:
        json.dump({0: "Safe", 1: "Defective"}, f)

if __name__ == "__main__":
    clean_and_tokenize()
