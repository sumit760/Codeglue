#!/usr/bin/env python3
"""
inference.py — Classify C/C++ code snippets as CLEAN or VULNERABLE using
the fine-tuned model hosted on the Hugging Face Hub.

Usage examples:
    # Run the built-in demo suite
    python inference.py

    # Classify a single snippet passed on the command line
    python inference.py --snippet "char buf[10]; gets(buf);"

    # Read a snippet from a file
    python inference.py --file path/to/snippet.c

    # Use a local model directory instead of the Hub
    python inference.py --model ./outputs/v2
"""

import argparse
import os
import sys

import torch
from huggingface_hub import login
from transformers import pipeline

from utils import HF_REPO_ID, MAX_LENGTH


# ── Built-in demo snippets ────────────────────────────────────────────────────

DEMO_SNIPPETS = [
    # (code, expected_label)
    ("char buf[10]; gets(buf);",                                "VULNERABLE"),
    ("printf(user_input);",                                     "VULNERABLE"),
    ("memcpy(dst, src, strlen(src));",                          "VULNERABLE"),
    ("if (size > MAX) { return -EINVAL; }",                     "VULNERABLE"),
    ("strncpy(dst, src, sizeof(dst) - 1);",                     "CLEAN"),
    ("ptr = malloc(n); if (!ptr) { return -ENOMEM; }",          "CLEAN"),
    ("snprintf(buf, sizeof(buf), \"%s\", user_input);",         "CLEAN"),
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def build_classifier(model_path: str):
    """Load the text-classification pipeline from the Hub or a local path."""
    if os.environ.get("HF_TOKEN"):
        login(token=os.environ["HF_TOKEN"])
    return pipeline(
        "text-classification",
        model=model_path,
        device=0 if torch.cuda.is_available() else -1,
    )


def classify(clf, code: str) -> dict:
    result = clf(code, truncation=True, max_length=MAX_LENGTH)[0]
    return {"label": result["label"], "confidence": result["score"]}


def run_demo(clf) -> None:
    print(f"\n{'Status':6}  {'Label':12}  {'Conf':6}  Snippet")
    print("-" * 72)
    passed = 0
    for snippet, expected in DEMO_SNIPPETS:
        out    = classify(clf, snippet)
        ok     = out["label"] == expected
        status = "✅ OK  " if ok else "❌ FAIL"
        passed += int(ok)
        print(f"{status}  {out['label']:12}  {out['confidence']:.4f}  {snippet[:55]}")
    print(f"\n{passed}/{len(DEMO_SNIPPETS)} demo snippets classified correctly.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="CodeBERTa vulnerability classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--model", default=HF_REPO_ID,
        help=f"HF repo ID or local path (default: {HF_REPO_ID})",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--snippet", type=str, default=None,
        help="Inline C/C++ code snippet to classify",
    )
    group.add_argument(
        "--file", type=str, default=None,
        help="Path to a .c / .cpp file to classify",
    )
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    print(f"Loading model from: {args.model}")
    clf  = build_classifier(args.model)

    if args.snippet:
        out = classify(clf, args.snippet)
        print(f"\nLabel      : {out['label']}")
        print(f"Confidence : {out['confidence']:.4f}")

    elif args.file:
        if not os.path.isfile(args.file):
            print(f"Error: file not found — {args.file}", file=sys.stderr)
            sys.exit(1)
        with open(args.file) as f:
            code = f.read()
        out = classify(clf, code)
        print(f"\nFile       : {args.file}")
        print(f"Label      : {out['label']}")
        print(f"Confidence : {out['confidence']:.4f}")

    else:
        run_demo(clf)


if __name__ == "__main__":
    main()
