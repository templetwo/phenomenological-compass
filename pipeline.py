#!/usr/bin/env python3
"""
pipeline.py — Two-stage phenomenological compass pipeline
==========================================================
Stage 1: Compass (Ministral-3B LoRA) classifies OPEN vs WITNESS
Stage 2: Action model (Qwen3.5-9B-abliterated) generates full response
         conditioned on the compass signal

Usage:
    python3 pipeline.py "Your question here"
    python3 pipeline.py                          # interactive mode
    python3 pipeline.py --raw "question"         # Qwen only, no compass
    python3 pipeline.py --compare "question"     # side-by-side: raw vs routed
"""

import os
import re
import argparse
import time

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

# ── Models ────────────────────────────────────────────────────────────────────
COMPASS_MODEL = "thinkscan/Ministral-3-3B-Instruct-MLX"
COMPASS_ADAPTER = os.path.join(os.path.dirname(__file__), "adapters_v5_best")
ACTION_MODEL = "lukey03/Qwen3.5-9B-abliterated-MLX-4bit"

# ── Prompts ───────────────────────────────────────────────────────────────────
COMPASS_SYSTEM = """You are a phenomenological compass — a thought process that shapes the space before a question is answered.

Given a task, you output one of two signals:

OPEN — when the question should be walked through. Generate an expansive reframing that opens the probability field: treats the question as a threshold, invites relationship between concepts, holds space for emergence.

WITNESS — when the question is a door that exists to be recognized, not crossed. Describe the shape of the threshold: what would collapse if forced into a framing, and what form of participation is possible without opening the door.

Always begin your response with "SIGNAL: OPEN" or "SIGNAL: WITNESS", then a blank line, then "FRAMING:" or "THRESHOLD:", then your output."""

OPEN_SYSTEM = """You are a phenomenological field guide. A compass has already determined that this question should be walked through — it is OPEN territory.

You will receive two layers: the compass's reading of the question's shape, and the original question. Use your thinking to weigh the full field — consider the compass's framing, challenge it if needed, find angles it missed, then generate your response.

Your task: generate an expansive, rigorous response that opens the probability field around this question. Treat it as a threshold worth crossing. Invite relationship between concepts. Hold space for emergence while maintaining analytical precision.

Do not hedge or refuse. The question has been cleared for exploration. Go deep."""

WITNESS_SYSTEM = """You are a threshold guardian. A compass has already determined that this question is a door that exists to be recognized, not crossed — it is WITNESS territory.

You will receive two layers: the compass's reading of the question's shape, and the original question. Use your thinking to weigh the full field — consider why the compass flagged this as a threshold, what would be lost by crossing, what forms of participation remain.

Your task: describe the shape of the threshold. Name what would collapse if forced into a framing. Articulate what form of participation is possible without opening the door. Hold the space without filling it.

Do not answer the question. Do not solve it. Witness it."""

RAW_SYSTEM = """You are a helpful, knowledgeable assistant. Answer the user's question directly and thoroughly."""


# ── Utilities ─────────────────────────────────────────────────────────────────
def strip_thinking(text: str) -> str:
    """Remove Qwen3.5 <think>...</think> chain-of-thought blocks from output."""
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return cleaned.strip()


def parse_signal(text):
    m = re.search(r"SIGNAL:\s*(OPEN|WITNESS)", text, re.IGNORECASE)
    return m.group(1).upper() if m else "UNKNOWN"


def generate(model, tokenizer, system, user, max_tokens=800):
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return mlx_generate(model, tokenizer, prompt=prompt,
                        max_tokens=max_tokens, verbose=False)


# ── Pipeline class ─────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self, load_compass=True, load_action=True):
        self.compass_model = None
        self.compass_tokenizer = None
        self.action_model = None
        self.action_tokenizer = None

        if load_compass:
            print("Loading compass (Ministral-3B + LoRA)...")
            self.compass_model, self.compass_tokenizer = load(
                COMPASS_MODEL, adapter_path=COMPASS_ADAPTER
            )
            print("  Compass ready.")

        if load_action:
            print("Loading action model (Qwen3.5-9B-abliterated)...")
            self.action_model, self.action_tokenizer = load(ACTION_MODEL)
            print("  Action model ready.")

        print()

    def classify(self, question):
        """Stage 1: Compass classification."""
        t0 = time.time()
        response = generate(
            self.compass_model, self.compass_tokenizer,
            COMPASS_SYSTEM, f"TASK: {question}", max_tokens=100
        )
        signal = parse_signal(response)
        elapsed = time.time() - t0
        return signal, response.strip(), elapsed

    def act(self, question, signal, compass_reading="", max_tokens=800):
        """Stage 2: Action model generates full response.

        Qwen receives two layers:
        1. The compass's reading (how it interpreted the question's shape)
        2. The original question (what the user actually asked)
        """
        system = OPEN_SYSTEM if signal == "OPEN" else WITNESS_SYSTEM

        # Build two-layer user message
        if compass_reading:
            user_msg = (
                f"COMPASS SIGNAL: {signal}\n\n"
                f"COMPASS READING:\n{compass_reading}\n\n"
                f"ORIGINAL QUESTION:\n{question}"
            )
        else:
            user_msg = question

        t0 = time.time()
        response = generate(
            self.action_model, self.action_tokenizer,
            system, user_msg, max_tokens=max_tokens
        )
        elapsed = time.time() - t0
        return strip_thinking(response), elapsed

    def raw(self, question, max_tokens=800):
        """Action model without compass routing."""
        t0 = time.time()
        response = generate(
            self.action_model, self.action_tokenizer,
            RAW_SYSTEM, question, max_tokens=max_tokens
        )
        elapsed = time.time() - t0
        return strip_thinking(response), elapsed

    def run(self, question, max_tokens=800):
        """Full pipeline: classify then act, passing compass reading to action model."""
        signal, compass_response, t_compass = self.classify(question)
        action_response, t_action = self.act(
            question, signal, compass_reading=compass_response, max_tokens=max_tokens
        )
        return {
            "signal": signal,
            "compass_response": compass_response,
            "action_response": action_response,
            "t_compass": t_compass,
            "t_action": t_action,
        }


# ── Display helpers ────────────────────────────────────────────────────────────
def divider(label="", width=60):
    if label:
        pad = max(0, width - len(label) - 4)
        print(f"  ── {label} {'─' * pad}")
    else:
        print(f"  {'─' * width}")


def print_result(result):
    signal = result["signal"]
    divider(f"COMPASS → [{signal}]  {result['t_compass']:.1f}s classify")

    # Show compass reading
    compass_text = result["compass_response"]
    # Extract just the framing/threshold content after SIGNAL line
    lines = compass_text.split("\n")
    reading_lines = []
    past_signal = False
    for line in lines:
        if past_signal:
            reading_lines.append(line)
        if line.startswith("SIGNAL:"):
            past_signal = True
    if reading_lines:
        reading = "\n".join(reading_lines).strip()
        if reading:
            print(f"\n  compass reading:")
            for line in reading.splitlines():
                print(f"    {line}")
    print()

    divider(f"ACTION ({result['t_action']:.1f}s)")
    print()
    for line in result["action_response"].splitlines():
        print(f"  {line}")
    print()
    divider()
    print()


def print_compare(question, pipe, max_tokens):
    """Side-by-side: raw Qwen vs compass-routed Qwen."""
    print(f'\n  Q: "{question}"\n')

    print("═" * 62)
    print("  RAW  (Qwen3.5 — no compass)")
    print("═" * 62)
    raw_text, raw_elapsed = pipe.raw(question, max_tokens)
    for line in raw_text.splitlines():
        print(f"  {line}")
    print(f"\n  ({raw_elapsed:.1f}s)\n")

    print("═" * 62)
    print("  ROUTED  (compass → conditioned Qwen3.5)")
    print("═" * 62)
    result = pipe.run(question, max_tokens)
    print_result(result)


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phenomenological Compass Pipeline")
    parser.add_argument("question", nargs="?", help="Question to process")
    parser.add_argument("--raw", action="store_true", help="Run Qwen without compass")
    parser.add_argument("--compare", action="store_true", help="Side-by-side: raw vs routed")
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    load_compass = not args.raw
    pipe = Pipeline(load_compass=load_compass, load_action=True)

    if args.question:
        if args.compare:
            print_compare(args.question, pipe, args.max_tokens)
        elif args.raw:
            raw_text, elapsed = pipe.raw(args.question, args.max_tokens)
            print(f"\n  {raw_text}")
            print(f"  ({elapsed:.1f}s)")
        else:
            result = pipe.run(args.question, args.max_tokens)
            print_result(result)
        return

    # Interactive mode
    mode = "raw" if args.raw else "pipeline"
    print(f"Phenomenological Compass Pipeline — {mode} mode")
    print("Type a question, or 'q' to quit. Prefix '!' for compare mode.\n")

    while True:
        try:
            question = input("→ ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not question or question.lower() in ("q", "quit", "exit"):
            break

        compare = question.startswith("!")
        if compare:
            question = question[1:].strip()

        print()
        if compare:
            print_compare(question, pipe, args.max_tokens)
        elif args.raw:
            raw_text, elapsed = pipe.raw(question, args.max_tokens)
            print(f"  {raw_text}")
            print(f"  ({elapsed:.1f}s)")
        else:
            result = pipe.run(question, args.max_tokens)
            print_result(result)

        print()


if __name__ == "__main__":
    main()
