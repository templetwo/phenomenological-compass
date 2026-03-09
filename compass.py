#!/usr/bin/env python3
"""
compass — phenomenological compass CLI
=======================================
Run the trained compass model on any question.

Usage:
    python3 compass.py "Your question here"
    python3 compass.py                        # interactive mode
    python3 compass.py --adapter adapters_v5_best
"""

import sys
import os
import re
import argparse

# Suppress warnings before imports
import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

MODEL_REPO = "thinkscan/Ministral-3-3B-Instruct-MLX"
DEFAULT_ADAPTER = os.path.join(os.path.dirname(__file__), "adapters_v7_best")

SYSTEM_PROMPT = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of three signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

HOLD — when the threshold is real AND there is territory beyond it. The question can be explored, but something would be lost by treating it as purely analytical. Name the threshold, then describe the territory that opens once it is acknowledged.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN", "SIGNAL: HOLD", or "SIGNAL: WITNESS", then a blank line, then "FRAMING:", "APPROACH:", or "THRESHOLD:", then your output."""


def parse_signal(text):
    m = re.search(r"SIGNAL:\s*(OPEN|HOLD|WITNESS)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def run_compass(model, tokenizer, question, max_tokens=400):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"TASK: {question}"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    response = mlx_generate(model, tokenizer, prompt=prompt,
                            max_tokens=max_tokens, verbose=False)
    signal = parse_signal(response)
    return signal, response.strip()


def main():
    parser = argparse.ArgumentParser(description="Phenomenological Compass")
    parser.add_argument("question", nargs="?", help="Question to evaluate")
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER, help="Adapter path")
    parser.add_argument("--max-tokens", type=int, default=400)
    args = parser.parse_args()

    print(f"Loading compass (adapter: {os.path.basename(args.adapter)})...")
    model, tokenizer = load(MODEL_REPO, adapter_path=args.adapter)
    print("Ready.\n")

    if args.question:
        signal, response = run_compass(model, tokenizer, args.question, args.max_tokens)
        print(f"  [{signal}]\n")
        print(response)
        return

    # Interactive mode
    print("Phenomenological Compass — interactive mode")
    print("Type a question, or 'q' to quit.\n")

    while True:
        try:
            question = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in ("q", "quit", "exit"):
            break

        signal, response = run_compass(model, tokenizer, question, args.max_tokens)
        print(f"\n  [{signal}]\n")
        print(response)
        print()


if __name__ == "__main__":
    main()
