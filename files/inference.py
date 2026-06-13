import argparse
import os
import sys

import torch
from huggingface_hub import login
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

from utils import HF_REPO_ID, MAX_LENGTH

DEMO_SNIPPETS = [
    ("char buf[10]; gets(buf);",                               "VULNERABLE"),
    ("printf(user_input);",                                    "VULNERABLE"),
    ("memcpy(dst, src, strlen(src));",                         "VULNERABLE"),
    ("strncpy(dst, src, sizeof(dst) - 1);",                    "CLEAN"),
    ("ptr = malloc(n); if (!ptr) { return -ENOMEM; }",         "CLEAN"),
]


def build_classifier(model_path):
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
    return pipeline(
        "text-classification",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
    )


def classify(clf, code):
    out = clf(code, truncation=True, max_length=MAX_LENGTH)[0]
    return {"label": out["label"], "confidence": out["score"]}


def run_demo(clf):
    print(f"{'Status':6}  {'Label':12}  {'Conf':6}  Snippet")
    print("-" * 70)
    for snippet, expected in DEMO_SNIPPETS:
        out    = classify(clf, snippet)
        status = "OK  " if out["label"] == expected else "FAIL"
        print(f"{status:6}  {out['label']:12}  {out['confidence']:.4f}  {snippet[:55]}")


def parse_args():
    p = argparse.ArgumentParser(description="CodeBERTa vulnerability classifier")
    p.add_argument("--model",   default=HF_REPO_ID)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--snippet", type=str, default=None)
    group.add_argument("--file",    type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"Loading model from: {args.model}")
    clf  = build_classifier(args.model)

    if args.snippet:
        out = classify(clf, args.snippet)
        print(f"Label      : {out['label']}")
        print(f"Confidence : {out['confidence']:.4f}")

    elif args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found - {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file) as f:
            code = f.read()
        out = classify(clf, code)
        print(f"File       : {args.file}")
        print(f"Label      : {out['label']}")
        print(f"Confidence : {out['confidence']:.4f}")

    else:
        run_demo(clf)


if __name__ == "__main__":
    main()
