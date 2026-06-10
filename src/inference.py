import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def run_inference(code_snippet):
    # CHANGE THIS TO YOUR ACTUAL HUGGING FACE REPO URL
    hf_repo = "your-username/your-model-repo" 
    
    try:
        model = AutoModelForSequenceClassification.from_pretrained(hf_repo)
        tokenizer = AutoTokenizer.from_pretrained(hf_repo)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    cleaned_code = " ".join(code_snippet.split())
    inputs = tokenizer(cleaned_code, return_tensors="pt", truncation=True, padding="max_length", max_length=512)

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        
    prediction = {0: "Safe", 1: "Defective"}[outputs.logits.argmax(dim=1).item()]
    print(f"\nInput Code Length: {len(code_snippet)} | Prediction: >> {prediction.upper()} <<\n")

if __name__ == "__main__":
    sample_code = "int main() { char buffer[10]; strcpy(buffer, 'too long'); return 0; }"
    run_inference(sample_code)
