"""
inference.py
============
Classifies a single code snippet as CLEAN or VULNERABLE using the model
published to the Hugging Face Hub.

Designed for the GitHub Actions `Inference` workflow and for the Docker
image: it reads its input entirely from environment variables, prints a
human-readable line plus a JSON object to stdout, and (when running in
Actions) writes a short GitHub step summary.

Environment variables
---------------------
INPUT_TEXT     required — the code snippet / text to classify
HF_TOKEN       optional — needed only if the model repo is private
HF_MODEL_NAME  optional — model repo on the Hub (set by the Dockerfile);
               falls back to HF_REPO_ID, then sumitp76/codeberta-vuln-detector
MAX_LENGTH     optional — tokenizer truncation length (default: 512)

Exit codes: 0 on success, 1 on bad/missing input, 2 on load/inference failure.
"""

import os
import re
import sys
import json

# Light imports only at module load. Heavy ML libraries (torch/transformers)
# are imported inside classify() *after* input validation, so a missing
# INPUT_TEXT fails fast without paying the import cost.

# Model repo: HF_MODEL_NAME is the primary name (set by the Dockerfile);
# HF_REPO_ID is accepted as a fallback for parity with train.py.
HF_MODEL_NAME = (
    os.environ.get("HF_MODEL_NAME")
    or os.environ.get("HF_REPO_ID")
    or "sumitp76/codeberta-vuln-detector"
)
HF_TOKEN = os.environ.get("HF_TOKEN") or None
INPUT_TEXT = os.environ.get("INPUT_TEXT")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "512"))


def clean_text(code: str) -> str:
    """Mirror data_prep.clean_text so serving preprocessing matches training.

    Kept inline (rather than imported from data_prep) so the inference image
    needs only torch + transformers, not datasets/sklearn/wandb.
    """
    code = re.sub(r"[ \t]+", " ", code)
    code = re.sub(r"\n{3,}", "\n\n", code)
    return code.strip()


def _write_github_summary(label: str, score: float) -> None:
    """If running inside GitHub Actions, append a small result summary."""
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not summary_path:
        return
    try:
        with open(summary_path, "a") as f:
            f.write("### Code Vulnerability Classification\n\n")
            f.write(f"- **Prediction:** `{label}`\n")
            f.write(f"- **Confidence:** {score:.4f}\n")
            f.write(f"- **Model:** `{HF_MODEL_NAME}`\n")
    except OSError:
        pass  # summary is best-effort; never fail the run over it


def classify(text: str) -> dict:
    """Load the published model and classify a single snippet."""
    import torch
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification, pipeline,
    )

    print(f"Loading model from '{HF_MODEL_NAME}' …")
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_NAME, token=HF_TOKEN)
    model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_NAME, token=HF_TOKEN
    )

    classifier = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=0 if torch.cuda.is_available() else -1,
    )

    out = classifier(clean_text(text), truncation=True, max_length=MAX_LENGTH)[0]
    return {
        "model": HF_MODEL_NAME,
        "label": out["label"],
        "score": round(float(out["score"]), 4),
        "input": text,
    }


def main() -> int:
    if not INPUT_TEXT or not INPUT_TEXT.strip():
        print("ERROR: INPUT_TEXT environment variable is required and non-empty.",
              file=sys.stderr)
        return 1

    try:
        result = classify(INPUT_TEXT)
    except Exception as exc:  # noqa: BLE001 - surface any load/inference failure clearly
        print(f"ERROR: inference failed: {exc}", file=sys.stderr)
        return 2

    print(f"\nPrediction: {result['label']}  (confidence {result['score']:.4f})")
    print(json.dumps(result, indent=2))
    _write_github_summary(result["label"], result["score"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
