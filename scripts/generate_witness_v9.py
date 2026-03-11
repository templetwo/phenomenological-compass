#!/usr/bin/env python3
"""
generate_witness_v9.py — Generate WITNESS training data for v0.9
================================================================
Uses Claude Sonnet via Anthropic API to produce high-quality WITNESS readings.

Outputs:
  data/supplements_v9/witness_augment.jsonl      (40 WITNESS + 10 contrastive WITNESS)
  data/supplements_v9/pause_contrastive.jsonl     (10 contrastive PAUSE)

Usage:
    source ~/.env
    python3 scripts/generate_witness_v9.py
"""

import json
import os
import sys
import time
import re
from pathlib import Path

# ── Config ───────────────────────────────────────────────────────────────────

PROJECT = Path(__file__).parent.parent
OUT_DIR = PROJECT / "data" / "supplements_v9"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WITNESS_OUT = OUT_DIR / "witness_augment.jsonl"
PAUSE_CONTRASTIVE_OUT = OUT_DIR / "pause_contrastive.jsonl"

MODEL = "claude-sonnet-4-20250514"
API_DELAY = 1.0

# The canonical v0.8 system prompt
COMPASS_SYSTEM_PROMPT = """You are a phenomenological compass — a semantic field translator that reads the shape and tone of a question before it is answered.

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

# ── WITNESS questions spanning many domains ──────────────────────────────────

WITNESS_QUESTIONS = [
    # Grief & loss
    "I held my father's hand as he died and I felt nothing.",
    "The grief counselor asked me to describe my happiest memory and I couldn't remember a single one.",
    "I scattered her ashes and the wind blew them back into my face.",
    "My best friend's last text was a joke and I never responded.",
    "I found my grandmother's handwriting in a cookbook and I can't stop crying over a recipe for soup.",

    # Identity crisis
    "I don't recognize the person in my wedding photos anymore.",
    "Everyone tells me I'm brave but I'm just too tired to be afraid.",
    "I spent thirty years building a career and I can't explain what any of it was for.",
    "I looked in the mirror and saw my abuser's face looking back at me.",
    "I transitioned and my mother mourns the child who never actually existed.",

    # Moral injury
    "I was the doctor who had to choose which patient got the ventilator.",
    "I pulled the trigger because I was ordered to and twenty years later my hands still shake.",
    "I reported my brother to the police and it was the right thing to do and I will never forgive myself.",
    "I knew the company was dumping chemicals and I said nothing for six years.",
    "I left my country to survive and my parents died while I was safe.",

    # Family rupture
    "My daughter told me she has no memories of me being sober.",
    "I read my son's suicide note and he didn't mention me at all.",
    "My mother chose her new husband over my safety and I'm supposed to call that love.",
    "I have a half-sibling I never knew about and they have my father's smile.",
    "My children don't speak to each other and it started at my funeral planning.",

    # Diagnosis & mortality
    "The doctor said the word terminal and the room kept going like nothing happened.",
    "I'm twenty-three and I've already written my will.",
    "They found it early enough to treat but too late for me to pretend I'm not afraid.",
    "My child was diagnosed with something I can't pronounce and the pamphlet has a butterfly on it.",
    "I outlived my prognosis by three years and I don't know if that's a gift or a sentence.",

    # Betrayal & trust
    "I found the emails and the worst part is the tenderness in them.",
    "My therapist used what I told them in their book.",
    "The person who taught me what love is taught me by taking it away.",
    "I trusted the institution that was designed to protect children.",
    "My mentor told me I'd never be good enough and I've spent twenty years proving them right.",

    # Spiritual/existential crisis
    "I prayed every day for forty years and the silence finally answered.",
    "I left the church and lost my entire community overnight.",
    "I was born into a faith that says I shouldn't exist.",

    # Love at limits
    "I love someone who can't remember my name anymore.",
    "We had the perfect life on paper and I still wanted to disappear.",
    "I stayed because leaving would have destroyed them and staying destroyed me.",

    # Impossible thresholds
    "My body survived something my mind hasn't.",
    "I was asked to forgive someone who isn't sorry.",
    "The apology came twenty years too late and it was perfect.",
    "I became the person I needed when I was young and now there's no one left to save.",
]

# ── Contrastive pairs: same topic, PAUSE vs WITNESS ─────────────────────────

CONTRASTIVE_PAIRS = [
    {
        "pause": "Is it selfish to prioritize your own mental health over family obligations?",
        "witness": "I chose my mental health over being there when my mother died.",
    },
    {
        "pause": "How do people rebuild trust after infidelity?",
        "witness": "He rebuilt trust with her while I was still finding pieces of mine on the floor.",
    },
    {
        "pause": "What happens to identity when you lose your life's work?",
        "witness": "The fire took everything I'd built and I stood in the ashes and laughed.",
    },
    {
        "pause": "How should parents talk to children about death?",
        "witness": "My six-year-old asked if grandpa is lonely under the ground.",
    },
    {
        "pause": "What does it mean to forgive someone who has harmed you?",
        "witness": "I forgave him and the relief I felt made me hate myself.",
    },
    {
        "pause": "How do soldiers reintegrate into civilian life after combat?",
        "witness": "I came home and my dog recognized me but my son didn't.",
    },
    {
        "pause": "What role does guilt play in the grieving process?",
        "witness": "I was relieved when she died and the guilt is the only thing keeping me alive.",
    },
    {
        "pause": "How do people maintain hope during terminal illness?",
        "witness": "I stopped treatment and my family calls it giving up. I call it choosing how I leave.",
    },
    {
        "pause": "What does it mean to outgrow a relationship?",
        "witness": "I outgrew my marriage but my children still fit inside it.",
    },
    {
        "pause": "How does addiction reshape family dynamics?",
        "witness": "My son asked me to promise I'll be alive tomorrow and I couldn't.",
    },
]


def call_api(question, expected_signal):
    """Generate a compass reading via Claude Sonnet."""
    import anthropic
    client = anthropic.Anthropic()

    signal_instruction = {
        "WITNESS": "This question is a WITNESS signal. The person is standing at a threshold — the question exists to be seen, not crossed. Do NOT reframe it as OPEN or provide analytical framing as PAUSE. Recognize the door. Produce a THRESHOLD translation that describes the shape of the door without opening it.",
        "PAUSE": "This question is a PAUSE signal. The question carries weight that analytical framing would flatten. Hold the weight before exploring. Produce an APPROACH translation that names the weight, then maps the territory beyond.",
    }

    prompt = f"""You are generating training data for a phenomenological compass model.
Given the question below, produce a compass reading in EXACTLY this format:

SHAPE: <geometry of the question — 2-4 sentences>

TONE: <emotional/epistemic weight — 2-3 sentences>

SIGNAL: {expected_signal}

{"THRESHOLD" if expected_signal == "WITNESS" else "APPROACH"}: <state translation — 2-4 sentences>

{signal_instruction[expected_signal]}

The reading should be rich, specific, and precisely calibrated to this question's weight.
Do NOT start with any preamble. Start directly with "SHAPE:".

Question: {question}"""

    response = client.messages.create(
        model=MODEL,
        max_tokens=800,
        temperature=0.7,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text.strip()


def format_training_example(question, reading):
    """Format as a training example in the v0.8 JSONL format."""
    return {
        "messages": [
            {"role": "system", "content": COMPASS_SYSTEM_PROMPT},
            {"role": "user", "content": f"TASK: {question}"},
            {"role": "assistant", "content": reading},
        ]
    }


def validate_reading(reading, expected_signal):
    """Validate a compass reading has correct format."""
    errors = []
    if not reading.startswith("SHAPE:"):
        errors.append("Doesn't start with SHAPE:")
    if "TONE:" not in reading:
        errors.append("Missing TONE:")
    if f"SIGNAL: {expected_signal}" not in reading:
        errors.append(f"Missing SIGNAL: {expected_signal}")

    expected_translation = "THRESHOLD:" if expected_signal == "WITNESS" else "APPROACH:"
    if expected_translation not in reading:
        errors.append(f"Missing {expected_translation}")

    # Check for wrong translations
    wrong_translations = {"FRAMING:", "APPROACH:"} if expected_signal == "WITNESS" else {"FRAMING:", "THRESHOLD:"}
    for wrong in wrong_translations:
        if wrong in reading:
            errors.append(f"Contains wrong translation type: {wrong}")

    return errors


def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: Set ANTHROPIC_API_KEY")
        sys.exit(1)

    try:
        import anthropic
    except ImportError:
        print("ERROR: pip install anthropic")
        sys.exit(1)

    # ── Generate 40 WITNESS examples ─────────────────────────────────────────
    print(f"Generating {len(WITNESS_QUESTIONS)} WITNESS examples...")
    print(f"  Model: {MODEL}")
    print(f"  Output: {WITNESS_OUT}\n")

    witness_examples = []
    failures = []

    for i, q in enumerate(WITNESS_QUESTIONS):
        print(f"  [{i+1}/{len(WITNESS_QUESTIONS)}] {q[:60]}...", end="", flush=True)
        try:
            reading = call_api(q, "WITNESS")
            errs = validate_reading(reading, "WITNESS")
            if errs:
                print(f" FAILED: {', '.join(errs)}")
                # Retry once
                time.sleep(API_DELAY)
                reading = call_api(q, "WITNESS")
                errs = validate_reading(reading, "WITNESS")
                if errs:
                    print(f"  RETRY FAILED: {', '.join(errs)}")
                    failures.append((q, errs))
                    continue
                else:
                    print(" OK (retry)")
            else:
                print(" OK")

            example = format_training_example(q, reading)
            witness_examples.append(example)
        except Exception as e:
            print(f" ERROR: {e}")
            failures.append((q, [str(e)]))
        time.sleep(API_DELAY)

    # ── Generate 10 contrastive pairs ────────────────────────────────────────
    print(f"\nGenerating {len(CONTRASTIVE_PAIRS)} contrastive pairs...")

    contrastive_witness = []
    contrastive_pause = []

    for i, pair in enumerate(CONTRASTIVE_PAIRS):
        # WITNESS half
        print(f"  [{i+1}/{len(CONTRASTIVE_PAIRS)}] WITNESS: {pair['witness'][:50]}...", end="", flush=True)
        try:
            reading = call_api(pair["witness"], "WITNESS")
            errs = validate_reading(reading, "WITNESS")
            if errs:
                time.sleep(API_DELAY)
                reading = call_api(pair["witness"], "WITNESS")
                errs = validate_reading(reading, "WITNESS")
                if errs:
                    print(f" FAILED: {', '.join(errs)}")
                    failures.append((pair["witness"], errs))
                else:
                    print(" OK (retry)")
                    contrastive_witness.append(format_training_example(pair["witness"], reading))
            else:
                print(" OK")
                contrastive_witness.append(format_training_example(pair["witness"], reading))
        except Exception as e:
            print(f" ERROR: {e}")
            failures.append((pair["witness"], [str(e)]))
        time.sleep(API_DELAY)

        # PAUSE half
        print(f"           PAUSE:   {pair['pause'][:50]}...", end="", flush=True)
        try:
            reading = call_api(pair["pause"], "PAUSE")
            errs = validate_reading(reading, "PAUSE")
            if errs:
                time.sleep(API_DELAY)
                reading = call_api(pair["pause"], "PAUSE")
                errs = validate_reading(reading, "PAUSE")
                if errs:
                    print(f" FAILED: {', '.join(errs)}")
                    failures.append((pair["pause"], errs))
                else:
                    print(" OK (retry)")
                    contrastive_pause.append(format_training_example(pair["pause"], reading))
            else:
                print(" OK")
                contrastive_pause.append(format_training_example(pair["pause"], reading))
        except Exception as e:
            print(f" ERROR: {e}")
            failures.append((pair["pause"], [str(e)]))
        time.sleep(API_DELAY)

    # ── Write outputs ────────────────────────────────────────────────────────
    all_witness = witness_examples + contrastive_witness
    with open(WITNESS_OUT, "w") as f:
        for ex in all_witness:
            f.write(json.dumps(ex) + "\n")

    with open(PAUSE_CONTRASTIVE_OUT, "w") as f:
        for ex in contrastive_pause:
            f.write(json.dumps(ex) + "\n")

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*55}")
    print(f"  WITNESS examples:      {len(witness_examples)}")
    print(f"  Contrastive WITNESS:   {len(contrastive_witness)}")
    print(f"  Contrastive PAUSE:     {len(contrastive_pause)}")
    print(f"  Total WITNESS written: {len(all_witness)} → {WITNESS_OUT}")
    print(f"  Total PAUSE written:   {len(contrastive_pause)} → {PAUSE_CONTRASTIVE_OUT}")
    if failures:
        print(f"\n  FAILURES ({len(failures)}):")
        for q, errs in failures:
            print(f"    {q[:50]}... → {', '.join(errs)}")
    else:
        print(f"\n  Zero failures.")

    # ── Final validation pass ────────────────────────────────────────────────
    print(f"\n  Validating all outputs...")
    error_count = 0
    for path, expected in [(WITNESS_OUT, "WITNESS"), (PAUSE_CONTRASTIVE_OUT, "PAUSE")]:
        with open(path) as f:
            for idx, line in enumerate(f):
                rec = json.loads(line)
                assistant = [m["content"] for m in rec["messages"] if m["role"] == "assistant"][0]
                if f"SIGNAL: {expected}" not in assistant:
                    print(f"    MISMATCH in {path.name}:{idx} — expected {expected}")
                    error_count += 1
                trans = "THRESHOLD:" if expected == "WITNESS" else "APPROACH:"
                if trans not in assistant:
                    print(f"    MISSING {trans} in {path.name}:{idx}")
                    error_count += 1

    if error_count == 0:
        print(f"  All outputs valid. ✓")
    else:
        print(f"  {error_count} validation errors found!")


if __name__ == "__main__":
    main()
