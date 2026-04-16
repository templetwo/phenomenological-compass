#!/usr/bin/env python3
"""
pipeline.py — v0.9 Phenomenological compass pipeline
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

import mlx.core as mx
from mlx_lm.utils import load
from mlx_lm.generate import generate as mlx_generate, stream_generate

# Sovereign Stack — read-only chronicle context
try:
    from stack_reader import is_available as stack_available, get_context_for_question as stack_context
    STACK_CONNECTED = stack_available()
except ImportError:
    STACK_CONNECTED = False
    def stack_context(q, **kw): return ""

# ── Models ────────────────────────────────────────────────────────────────────
COMPASS_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
COMPASS_ADAPTER = os.path.join(os.path.dirname(__file__), "adapters_v10_qwen")
COMPASS_CHECKPOINT = "0000500_adapters.safetensors"  # iter 500: 84% boundary, 100% real-world

ACTION_MODELS = {
    "gemma-e2b": {
        "ollama": "gemma4:e2b",
        "name": "Gemma4-E2B",
        "engine": "ollama",
        "has_thinking": False,
    },
    "gemma-8b": {
        "ollama": "gemma4:latest",
        "name": "Gemma4-8B",
        "engine": "ollama",
        "has_thinking": False,
    },
    "gemma-26b": {
        "ollama": "gemma4:26b",
        "name": "Gemma4-26B",
        "engine": "ollama",
        "has_thinking": False,
    },
    "qwen": {
        "repo": "lukey03/Qwen3.5-9B-abliterated-MLX-4bit",
        "name": "Qwen3.5-9B-abliterated",
        "engine": "mlx",
        "has_thinking": True,
    },
    "m14b": {
        "repo": "McG-221/Ministral-3-14B-abliterated-mlx-8Bit",
        "name": "Ministral-14B-abliterated",
        "engine": "mlx",
        "has_thinking": False,
    },
}
DEFAULT_ACTION = "gemma-e2b"

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
  If WITNESS → THRESHOLD: describe the shape of the door without opening it

Finally, your resource allocation. The responding model has a fixed token budget. You decide how it spends that budget based on the signal:

BUDGET — output two numbers (thinking: N, response: N) that sum to 800:
  If OPEN → thinking: 200, response: 600 (explore the territory, then walk through it fully)
  If PAUSE → thinking: 150, response: 650 (feel the weight briefly, then meet it with depth)
  If WITNESS → thinking: 50, response: 750 (you already know what to do — spend everything on presence)
  Adjust these defaults based on complexity. A simple OPEN question might need thinking: 100, response: 700. A deeply recursive PAUSE might need thinking: 250, response: 550. Trust your reading."""

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
    """Extract signal from compass reading. Falls back to PAUSE if unparseable."""
    m = re.search(r"SIGNAL:\s*(OPEN|PAUSE|WITNESS)", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Fallback: try to detect signal words anywhere in text
    text_upper = text.upper()
    if "WITNESS" in text_upper and "THRESHOLD" in text_upper:
        return "WITNESS"
    if "PAUSE" in text_upper and ("WEIGHT" in text_upper or "APPROACH" in text_upper):
        return "PAUSE"
    if "OPEN" in text_upper and ("FRAMING" in text_upper or "EXPLORE" in text_upper):
        return "OPEN"
    # Safe default: PAUSE (holds weight without being evasive or overconfident)
    return "PAUSE"


def parse_budget(text, signal="OPEN"):
    """Extract token budget from compass reading. Returns (thinking, response) tuple."""
    # Try to find explicit budget
    m = re.search(r"thinking:\s*(\d+).*?response:\s*(\d+)", text, re.IGNORECASE | re.DOTALL)
    if m:
        return int(m.group(1)), int(m.group(2))
    # Default budgets by signal
    defaults = {
        "OPEN": (200, 600),
        "PAUSE": (150, 650),
        "WITNESS": (50, 750),
    }
    return defaults.get(signal, (200, 600))


def generate(model, tokenizer, system, user, max_tokens=2048):
    """Generate with generous ceiling. Let EOS handle natural stopping."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    result = mlx_generate(model, tokenizer, prompt=prompt,
                          max_tokens=max_tokens, verbose=False)
    return result


# ── Ollama Generation ────────────────────────────────────────────────────────
import requests as _requests
import json as _json

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")


def ollama_generate(model_name, system, user, max_tokens=2048):
    """Generate via Ollama API. Returns response text."""
    resp = _requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "num_predict": max_tokens,
            "num_ctx": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.0,
        },
        "stream": False,
    }, timeout=300)
    resp.raise_for_status()
    return resp.json()["message"]["content"]


def ollama_stream(model_name, system, user, max_tokens=2048):
    """Stream via Ollama API. Yields (text, None, finish_reason, tps) per chunk."""
    resp = _requests.post(f"{OLLAMA_URL}/api/chat", json={
        "model": model_name,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "options": {
            "num_predict": max_tokens,
            "num_ctx": 8192,
            "temperature": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.0,
        },
        "stream": True,
    }, timeout=300, stream=True)
    resp.raise_for_status()
    total_tokens = 0
    t0 = time.time()
    for line in resp.iter_lines():
        if not line:
            continue
        data = _json.loads(line)
        msg = data.get("message", {})
        text = msg.get("content", "")
        done = data.get("done", False)
        if text:
            total_tokens += 1
            elapsed = time.time() - t0
            tps = total_tokens / elapsed if elapsed > 0 else 0
            finish = "stop" if done else None
            yield text, None, finish, round(tps, 1)
        if done:
            break


# ── Pipeline class ────────────────────────────────────────────────────────────
class Pipeline:
    def __init__(self, load_compass=True, load_action=True, action_key=None,
                 adapter_path=None, adapter_checkpoint=None):
        self.compass_model = None
        self.compass_tokenizer = None
        self.action_model = None
        self.action_tokenizer = None

        self.action_key = action_key or DEFAULT_ACTION
        self.action_config = ACTION_MODELS[self.action_key]

        # Allow override of adapter path/checkpoint
        _adapter_dir = adapter_path or COMPASS_ADAPTER
        _adapter_cp = adapter_checkpoint or COMPASS_CHECKPOINT

        if load_compass:
            version = "v0.9" if "v9" in str(_adapter_dir) else "v0.8"
            print(f"Loading compass (Ministral-3B + {version} LoRA from {os.path.basename(_adapter_dir)})...")
            # Ensure best checkpoint is active
            cp_path = os.path.join(_adapter_dir, _adapter_cp)
            active_path = os.path.join(_adapter_dir, "adapters.safetensors")
            if os.path.exists(cp_path):
                shutil.copy2(cp_path, active_path)
            self.compass_model, self.compass_tokenizer = load(
                COMPASS_MODEL, adapter_path=_adapter_dir
            )
            print("  Compass ready.")

        if load_action:
            name = self.action_config["name"]
            engine = self.action_config.get("engine", "mlx")
            if engine == "ollama":
                self.action_engine = "ollama"
                self.ollama_model = self.action_config["ollama"]
                print(f"  Action model: {name} via Ollama ({self.ollama_model})")
            else:
                self.action_engine = "mlx"
                print(f"Loading action model ({name})...")
                self.action_model, self.action_tokenizer = load(self.action_config["repo"])
                print(f"  Action model ready ({name}).")

        print()

    def classify(self, question):
        """Stage 1: Compass reads shape, tone, signal, and translation."""
        t0 = time.time()
        try:
            response = generate(
                self.compass_model, self.compass_tokenizer,
                COMPASS_SYSTEM, f"TASK: {question}", max_tokens=800
            )
        except Exception as e:
            # Compass failure: fall back to PAUSE with empty reading
            elapsed = time.time() - t0
            print(f"  [COMPASS ERROR: {e}] Falling back to PAUSE")
            return "PAUSE", "", elapsed
        signal = parse_signal(response)
        elapsed = time.time() - t0
        return signal, response.strip(), elapsed

    def act(self, question, signal, compass_reading="", max_tokens=2048):
        """Stage 2: Action model generates response conditioned on compass.

        Action model receives two layers:
        1. The compass's full state translation (SHAPE, TONE, SIGNAL, translation)
        2. The original question (what the user actually asked)
        """
        SIGNAL_SYSTEMS = {
            "OPEN": OPEN_SYSTEM,
            "PAUSE": PAUSE_SYSTEM,
            "WITNESS": WITNESS_SYSTEM,
        }
        system = SIGNAL_SYSTEMS.get(signal, PAUSE_SYSTEM)

        # Build multi-layer user message: Stack context + Compass reading + Question
        stack_ctx = stack_context(question, max_chars=600) if STACK_CONNECTED else ""

        layers = []
        if stack_ctx:
            layers.append(stack_ctx)
        if compass_reading:
            layers.append(f"COMPASS READING:\n{compass_reading}")
        layers.append(f"ORIGINAL QUESTION:\n{question}")
        user_msg = "\n\n".join(layers)

        t0 = time.time()
        try:
            if getattr(self, "action_engine", "mlx") == "ollama":
                response = ollama_generate(self.ollama_model, system, user_msg, max_tokens)
            else:
                response = generate(
                    self.action_model, self.action_tokenizer,
                    system, user_msg, max_tokens=max_tokens
                )
        except Exception as e:
            elapsed = time.time() - t0
            return f"[Generation error: {e}]", elapsed, ""
        elapsed = time.time() - t0
        if self.action_config["has_thinking"]:
            thinking, clean = split_thinking(response)
        else:
            thinking, clean = "", response.strip()
        return clean, elapsed, thinking

    def raw(self, question, max_tokens=2048):
        """Action model without compass routing."""
        t0 = time.time()
        try:
            if getattr(self, "action_engine", "mlx") == "ollama":
                response = ollama_generate(self.ollama_model, RAW_SYSTEM, question, max_tokens)
            else:
                response = generate(
                    self.action_model, self.action_tokenizer,
                    RAW_SYSTEM, question, max_tokens=max_tokens
                )
        except Exception as e:
            elapsed = time.time() - t0
            return f"[Generation error: {e}]", elapsed, ""
        elapsed = time.time() - t0
        if self.action_config["has_thinking"]:
            thinking, clean = split_thinking(response)
        else:
            thinking, clean = "", response.strip()
        return clean, elapsed, thinking

    def _build_prompt(self, model_tokenizer, system, user):
        """Build chat prompt string from system + user messages."""
        tokenizer = model_tokenizer
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def stream_classify(self, question):
        """Stage 1 streaming: yields (text, logprobs, finish_reason, gen_tps) per token."""
        prompt = self._build_prompt(
            self.compass_tokenizer, COMPASS_SYSTEM, f"TASK: {question}"
        )
        for resp in stream_generate(
            self.compass_model, self.compass_tokenizer,
            prompt=prompt, max_tokens=800
        ):
            yield resp.text, resp.logprobs, resp.finish_reason, resp.generation_tps

    def stream_act(self, question, signal, compass_reading="", max_tokens=2048):
        """Stage 2 streaming: yields (text, logprobs, finish_reason, gen_tps) per token."""
        SIGNAL_SYSTEMS = {
            "OPEN": OPEN_SYSTEM,
            "PAUSE": PAUSE_SYSTEM,
            "WITNESS": WITNESS_SYSTEM,
        }
        system = SIGNAL_SYSTEMS.get(signal, PAUSE_SYSTEM)

        # Build multi-layer user message (same as non-streaming act)
        stack_ctx = stack_context(question, max_chars=600) if STACK_CONNECTED else ""
        layers = []
        if stack_ctx:
            layers.append(stack_ctx)
        if compass_reading:
            layers.append(f"COMPASS READING:\n{compass_reading}")
        layers.append(f"ORIGINAL QUESTION:\n{question}")
        user_msg = "\n\n".join(layers)

        if getattr(self, "action_engine", "mlx") == "ollama":
            for item in ollama_stream(self.ollama_model, system, user_msg, max_tokens):
                yield item
        else:
            prompt = self._build_prompt(self.action_tokenizer, system, user_msg)
            for resp in stream_generate(
                self.action_model, self.action_tokenizer,
                prompt=prompt, max_tokens=max_tokens
            ):
                yield resp.text, resp.logprobs, resp.finish_reason, resp.generation_tps

    def stream_raw(self, question, max_tokens=800):
        """Raw streaming (no compass): yields (text, logprobs, finish_reason, gen_tps)."""
        if getattr(self, "action_engine", "mlx") == "ollama":
            for item in ollama_stream(self.ollama_model, RAW_SYSTEM, question, max_tokens):
                yield item
        else:
            prompt = self._build_prompt(self.action_tokenizer, RAW_SYSTEM, question)
            for resp in stream_generate(
                self.action_model, self.action_tokenizer,
                prompt=prompt, max_tokens=max_tokens
            ):
                yield resp.text, resp.logprobs, resp.finish_reason, resp.generation_tps

    def run_with_signal(self, question, forced_signal, max_tokens=800):
        """Bypass compass. Inject forced signal directly into action model conditioning."""
        action_response, t_action, thinking = self.act(
            question, forced_signal, compass_reading="", max_tokens=max_tokens
        )
        return {
            "signal": forced_signal,
            "compass_response": "",
            "action_response": action_response,
            "thinking": thinking,
            "t_compass": 0.0,
            "t_action": t_action,
        }

    def breathe(self, question, compass_response, signal, depth=1):
        """The deliberate gap — where the compass reflects on its own reading.
        
        The compass reads the question (Stage 1). Then, instead of immediately
        routing to the action model, it re-reads the question THROUGH its own
        reading. Each breath cycle lets the question transform in the light
        of the previous reading. The question changes shape when you stop
        measuring it and start listening to the measurement.
        
        depth: number of reflection cycles (default 1, max 3)
        
        Returns: (transformed_question, final_signal, full_reading, breath_log)
        """
        breath_log = [{
            "cycle": 0,
            "signal": signal,
            "reading_preview": compass_response[:150],
            "question": question,
        }]
        
        current_question = question
        current_signal = signal
        current_reading = compass_response
        
        for cycle in range(min(depth, 3)):
            # The compass reads the question through its own prior reading
            reflection_prompt = (
                f"The compass has already read this question and found:\n"
                f"SIGNAL: {current_signal}\n"
                f"READING: {current_reading[:300]}\n\n"
                f"Now re-read the ORIGINAL question through that reading. "
                f"Has the shape changed? Has the tone shifted? "
                f"Does the signal still hold, or does the question reveal "
                f"a different face when seen through the first reading?\n\n"
                f"ORIGINAL QUESTION: {question}\n\n"
                f"Produce your reading: SHAPE, TONE, SIGNAL, translation, BUDGET."
            )
            
            try:
                new_reading = generate(
                    self.compass_model, self.compass_tokenizer,
                    COMPASS_SYSTEM, reflection_prompt, max_tokens=500
                )
                new_signal = parse_signal(new_reading)
                
                breath_log.append({
                    "cycle": cycle + 1,
                    "signal": new_signal,
                    "signal_changed": new_signal != current_signal,
                    "reading_preview": new_reading[:150],
                    "question": question,
                })
                
                current_signal = new_signal
                current_reading = new_reading
                
            except Exception as e:
                breath_log.append({"cycle": cycle + 1, "error": str(e)})
                break
        
        return question, current_signal, current_reading, breath_log

    def run(self, question, max_tokens=2048, gap_ms=0, breath_depth=0):
        """Full pipeline: classify, [breathe], then act.
        
        gap_ms: simple pause between reading and response (default: 0).
        breath_depth: number of compass reflection cycles (default: 0).
            0 = current behavior (classify then act)
            1 = one reflection cycle (compass re-reads through its own reading)
            2-3 = deeper reflection (the question seen through multiple readings)
        
        The gap is not latency. It is the space where the reading
        acts on the question before the question reaches the model.
        """
        signal, compass_response, t_compass = self.classify(question)
        
        # The deliberate gap — if requested
        breath_log = None
        if breath_depth > 0:
            t_breath_start = time.time()
            question, signal, compass_response, breath_log = self.breathe(
                question, compass_response, signal, depth=breath_depth
            )
            t_compass += time.time() - t_breath_start
        elif gap_ms > 0:
            time.sleep(gap_ms / 1000.0)
        
        think_budget, resp_budget = parse_budget(compass_response, signal)
        action_response, t_action, thinking = self.act(
            question, signal, compass_reading=compass_response, max_tokens=max_tokens
        )
        result = {
            "signal": signal,
            "compass_response": compass_response,
            "action_response": action_response,
            "thinking": thinking,
            "t_compass": t_compass,
            "t_action": t_action,
            "budget": {"thinking": think_budget, "response": resp_budget},
        }
        if breath_log:
            result["breath_log"] = breath_log
        return result


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
    parser = argparse.ArgumentParser(description="Phenomenological Compass Pipeline v0.9")
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
    print(f"Phenomenological Compass Pipeline v0.9 — {mode} mode [{name}]")
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
