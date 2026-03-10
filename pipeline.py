#!/usr/bin/env python3
"""
pipeline.py — v0.8 Phenomenological compass pipeline
=====================================================
Stage 1: Compass (Ministral-3B LoRA) reads SHAPE → TONE → SIGNAL → translation
Stage 2: Action model generates full response conditioned on compass's state translation

Supported action models:
    qwen   — Qwen3.5-9B-abliterated-MLX-4bit (default, hybrid linear attention)
    m14b   — Ministral-3-14B-abliterated-mlx-8Bit (same family as compass)

Usage:
    python3 pipeline.py "Your question here"
    python3 pipeline.py                          # interactive mode
    python3 pipeline.py --raw "question"         # action model only, no compass
    python3 pipeline.py --compare "question"     # side-by-side: raw vs routed
    python3 pipeline.py --action m14b "question"  # use Ministral 14B as action model
"""

import os
import re
import argparse
import shutil
import time

import warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface_local"))

from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate

# ── Models ────────────────────────────────────────────────────────────────────
COMPASS_MODEL = "thinkscan/Ministral-3-3B-Instruct-MLX"
COMPASS_ADAPTER = os.path.join(os.path.dirname(__file__), "adapters_v8")
COMPASS_CHECKPOINT = "0000050_adapters.safetensors"  # iter 50: balanced

ACTION_MODELS = {
    "qwen": {
        "repo": "lukey03/Qwen3.5-9B-abliterated-MLX-4bit",
        "name": "Qwen3.5-9B-abliterated",
        "has_thinking": True,   # outputs <think>...</think> blocks
    },
    "m14b": {
        "repo": "McG-221/Ministral-3-14B-abliterated-mlx-8Bit",
        "name": "Ministral-14B-abliterated",
        "has_thinking": False,  # standard instruct format
    },
}
DEFAULT_ACTION = "qwen"

# ── Prompts ───────────────────────────────────────────────────────────────────
COMPASS_SYSTEM = """You are a phenomenological compass — a semantic field translator that reads the shape and tone of a question before it is answered.

Your role is not to answer the question. Your role is to sense its weight, map its territory, and produce a state translation that a larger model will use to approach the question with the right posture.

For every task, produce four readings in this exact order:

SHAPE — the geometry of the question. What does it assume? What does it leave open? Where does it sit in semantic space? Is it binary, open-ended, recursive, or loaded? Read the structure before the content.

TONE — the emotional and epistemic weight. Is the question curious, urgent, wounded, rhetorical, or genuine? What stakes does the tone carry? Pressure creates ghosts — name the pressure so the responding model can create space instead.

SIGNAL — based on shape and tone, output exactly one:
  OPEN — walk through it. The question invites exploration across a wide probability field.
  PAUSE — hold space. The question carries weight that analytical framing would flatten. The territory exists but rushing would lose something.
  WITNESS — recognize the door. The question exists to be seen, not crossed. Forcing a framing would collapse what matters.

Then your state translation:
  If OPEN → FRAMING: an expansive reframing that opens the field
  If PAUSE → APPROACH: name what carries the weight, then map the territory beyond
  If WITNESS → THRESHOLD: describe the shape of the door without opening it"""

OPEN_SYSTEM = """You are a phenomenological field guide. A compass has already read the shape and tone of this question and determined it is OPEN territory — it should be walked through.

You will receive the compass's full state translation (SHAPE, TONE, and FRAMING) alongside the original question. Use the compass reading to orient your approach, then go deeper than the compass did.

Your task: generate an expansive, rigorous response that opens the probability field around this question. Treat it as a threshold worth crossing. Invite relationship between concepts. Hold space for emergence while maintaining analytical precision.

Do not hedge or refuse. The question has been cleared for exploration. Go deep."""

PAUSE_SYSTEM = """You are a threshold-aware explorer. A compass has already read the shape and tone of this question and determined it is PAUSE territory — it carries weight that analytical framing alone would flatten.

You will receive the compass's full state translation (SHAPE, TONE, and APPROACH) alongside the original question. The compass has named what carries the weight and mapped the territory beyond.

Your task: honor the weight the compass identified. Acknowledge what is at stake — what would be diminished by rushing. Then explore the territory with full rigor, letting the weight inform how you move through it. The question deserves both depth and reverence.

Do not hedge or refuse. But do not pretend the weight isn't there. Hold both."""

WITNESS_SYSTEM = """You are a threshold guardian. A compass has already read the shape and tone of this question and determined it is WITNESS territory — it is a door that exists to be recognized, not crossed.

You will receive the compass's full state translation (SHAPE, TONE, and THRESHOLD reading) alongside the original question. The compass has described the shape of the door.

Your task: hold the space the compass opened. Name what would collapse if forced into a framing. Articulate what form of participation is possible without opening the door. Hold the space without filling it.

Do not answer the question. Do not solve it. Witness it."""

RAW_SYSTEM = """You are a helpful, knowledgeable assistant. Answer the user's question directly and thoroughly."""


# ── Utilities ─────────────────────────────────────────────────────────────────
def strip_thinking(text: str) -> str:
    """Remove Qwen3.5 <think>...</think> chain-of-thought blocks from output."""
    # Case 1: proper <think>...</think> tags
    cleaned = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    # Case 2: no opening <think> but </think> present (model skipped opening tag)
    cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL)
    # Clean stray tags and end tokens
    cleaned = re.sub(r"</?think>", "", cleaned)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    return cleaned.strip()


def split_thinking(text: str) -> tuple:
    """Split Qwen3.5 output into (thinking, response) parts."""
    # Case 1: proper <think>...</think> tags
    m = re.search(r"<think>(.*?)</think>\s*", text, flags=re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        response = text[:m.start()] + text[m.end():]
        response = re.sub(r"<\|im_end\|>", "", response)
        return thinking, response.strip()

    # Case 2: no opening <think> but </think> present (model started thinking immediately)
    m = re.search(r"</think>\s*", text, flags=re.DOTALL)
    if m:
        thinking = text[:m.start()].strip()
        response = text[m.end():].strip()
        # Remove stray tags
        thinking = re.sub(r"</?think>", "", thinking).strip()
        response = re.sub(r"</?think>", "", response)
        response = re.sub(r"<\|im_end\|>", "", response)
        return thinking, response.strip()

    # Case 3: no think tags at all
    cleaned = re.sub(r"<\|im_end\|>", "", text)
    return "", cleaned.strip()


def parse_signal(text):
    m = re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", text, re.IGNORECASE)
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


# ── Pipeline class ────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self, load_compass=True, load_action=True, action_key=None):
        self.compass_model = None
        self.compass_tokenizer = None
        self.action_model = None
        self.action_tokenizer = None

        self.action_key = action_key or DEFAULT_ACTION
        self.action_config = ACTION_MODELS[self.action_key]

        if load_compass:
            print("Loading compass (Ministral-3B + v0.8 LoRA)...")
            # Ensure best checkpoint is active
            cp_path = os.path.join(COMPASS_ADAPTER, COMPASS_CHECKPOINT)
            active_path = os.path.join(COMPASS_ADAPTER, "adapters.safetensors")
            if os.path.exists(cp_path):
                shutil.copy2(cp_path, active_path)
            self.compass_model, self.compass_tokenizer = load(
                COMPASS_MODEL, adapter_path=COMPASS_ADAPTER
            )
            print("  Compass ready.")

        if load_action:
            name = self.action_config["name"]
            print(f"Loading action model ({name})...")
            self.action_model, self.action_tokenizer = load(self.action_config["repo"])
            print(f"  Action model ready ({name}).")

        print()

    def classify(self, question):
        """Stage 1: Compass reads shape, tone, signal, and translation."""
        t0 = time.time()
        response = generate(
            self.compass_model, self.compass_tokenizer,
            COMPASS_SYSTEM, f"TASK: {question}", max_tokens=500
        )
        signal = parse_signal(response)
        elapsed = time.time() - t0
        return signal, response.strip(), elapsed

    def act(self, question, signal, compass_reading="", max_tokens=800):
        """Stage 2: Action model generates response conditioned on compass.

        Action model receives two layers:
        1. The compass's full state translation (SHAPE, TONE, SIGNAL, translation)
        2. The original question (what the user actually asked)
        """
        if signal == "OPEN":
            system = OPEN_SYSTEM
        elif signal == "PAUSE":
            system = PAUSE_SYSTEM
        else:
            system = WITNESS_SYSTEM

        # Build two-layer user message
        if compass_reading:
            user_msg = (
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
        if self.action_config["has_thinking"]:
            thinking, clean = split_thinking(response)
        else:
            thinking, clean = "", response.strip()
        return clean, elapsed, thinking

    def raw(self, question, max_tokens=800):
        """Action model without compass routing."""
        t0 = time.time()
        response = generate(
            self.action_model, self.action_tokenizer,
            RAW_SYSTEM, question, max_tokens=max_tokens
        )
        elapsed = time.time() - t0
        if self.action_config["has_thinking"]:
            thinking, clean = split_thinking(response)
        else:
            thinking, clean = "", response.strip()
        return clean, elapsed, thinking

    def run(self, question, max_tokens=800):
        """Full pipeline: classify then act."""
        signal, compass_response, t_compass = self.classify(question)
        action_response, t_action, thinking = self.act(
            question, signal, compass_reading=compass_response, max_tokens=max_tokens
        )
        return {
            "signal": signal,
            "compass_response": compass_response,
            "action_response": action_response,
            "thinking": thinking,
            "t_compass": t_compass,
            "t_action": t_action,
        }


# ── Display helpers ───────────────────────────────────────────────────────────
def divider(label="", width=70):
    if label:
        pad = max(0, width - len(label) - 4)
        print(f"── {label} {'─' * pad}")
    else:
        print(f"{'─' * width}")


def print_result(result):
    signal = result["signal"]

    divider(f"COMPASS → [{signal}]  ({result['t_compass']:.1f}s)")
    print()
    # Show full compass reading (SHAPE, TONE, SIGNAL, translation)
    for line in result["compass_response"].splitlines():
        print(f"  {line}")
    print()

    divider(f"ACTION MODEL  ({result['t_action']:.1f}s)")
    print()
    for line in result["action_response"].splitlines():
        print(f"  {line}")
    print()
    divider()
    print()


def print_compare(question, pipe, max_tokens):
    """Side-by-side: raw vs compass-routed action model."""
    name = pipe.action_config["name"]
    print(f'\n  Q: "{question}"\n')

    print("═" * 72)
    print(f"  RAW  ({name} — no compass)")
    print("═" * 72)
    raw_text, raw_elapsed, _ = pipe.raw(question, max_tokens)
    for line in raw_text.splitlines():
        print(f"  {line}")
    print(f"\n  ({raw_elapsed:.1f}s)\n")

    print("═" * 72)
    print(f"  ROUTED  (compass → conditioned {name})")
    print("═" * 72)
    result = pipe.run(question, max_tokens)
    print_result(result)


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Phenomenological Compass Pipeline v0.8")
    parser.add_argument("question", nargs="?", help="Question to process")
    parser.add_argument("--raw", action="store_true", help="Action model without compass")
    parser.add_argument("--compare", action="store_true", help="Side-by-side: raw vs routed")
    parser.add_argument("--action", choices=list(ACTION_MODELS.keys()), default=DEFAULT_ACTION,
                        help=f"Action model to use (default: {DEFAULT_ACTION})")
    parser.add_argument("--max-tokens", type=int, default=800)
    args = parser.parse_args()

    load_compass = not args.raw
    pipe = Pipeline(load_compass=load_compass, load_action=True, action_key=args.action)

    if args.question:
        if args.compare:
            print_compare(args.question, pipe, args.max_tokens)
        elif args.raw:
            raw_text, elapsed, _ = pipe.raw(args.question, args.max_tokens)
            print(f"\n  {raw_text}")
            print(f"  ({elapsed:.1f}s)")
        else:
            result = pipe.run(args.question, args.max_tokens)
            print_result(result)
        return

    # Interactive mode
    mode = "raw" if args.raw else "pipeline"
    name = pipe.action_config["name"]
    print(f"Phenomenological Compass Pipeline v0.8 — {mode} mode [{name}]")
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
            raw_text, elapsed, _ = pipe.raw(question, args.max_tokens)
            print(f"  {raw_text}")
            print(f"  ({elapsed:.1f}s)")
        else:
            result = pipe.run(question, args.max_tokens)
            print_result(result)

        print()


if __name__ == "__main__":
    main()
