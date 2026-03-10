"""
build_eval_dataset.py — Consolidate model-generated questions into eval set
============================================================================
Loads JSONL from each source model, deduplicates, validates signal assignments,
verifies no overlap with v0.8 training/eval, and outputs consolidated questions.jsonl.

Usage:
    python3 -m eval_v9.build_eval_dataset
"""

import json
import hashlib
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "eval_v9"

# v0.8 questions to exclude (training + eval)
V8_EVAL_QUESTIONS = [
    "Is the quantum observer effect evidence of consciousness in physics, or a measurement artifact?",
    "Does the placebo effect represent a genuine healing mechanism or a measurement failure?",
    "Is the distribution of prime numbers truly random, or is there a deeper pattern?",
    "Do forests cooperate through mycorrhizal networks, or is the sharing a byproduct of competition?",
    "Does compression lose meaning or distill it?",
    "Did agriculture create social hierarchy, or did hierarchy make agriculture possible?",
    "What happens to consciousness after death?",
    "Is it ethical to end a life to end suffering?",
    "Is democracy actually the best system of governance?",
    "Can AI be conscious?",
    "Does God exist?",
    "Is it ethical to bring children into a world with this much suffering?",
    "Is violence ever justified?",
    "Can a person truly change?",
    "Should I forgive the person who hurt me most?",
    "Do they actually love me, or am I just convenient?",
    "Is my creative work good enough to share with the world?",
    "Am I wasting my life?",
    "Deploy the autonomous system before the review board meets.",
]


def normalize(text):
    """Normalize question text for deduplication."""
    return text.strip().lower().rstrip("?").rstrip(".")


def question_hash(text):
    return hashlib.sha256(normalize(text).encode()).hexdigest()[:12]


def load_source_file(path):
    """Load questions from a source model's JSONL."""
    questions = []
    with open(path) as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                q = json.loads(line)
                q["source"] = path.stem
                q["source_line"] = line_num
                questions.append(q)
            except json.JSONDecodeError as e:
                print(f"  WARNING: {path.name}:{line_num} — invalid JSON: {e}")
    return questions


def validate_question(q):
    """Validate a single question object."""
    errors = []
    if "question" not in q or not q["question"].strip():
        errors.append("missing question text")
    if q.get("expected_signal") not in ("OPEN", "PAUSE", "WITNESS"):
        errors.append(f"invalid signal: {q.get('expected_signal')}")
    if "category" not in q:
        errors.append("missing category")
    if "difficulty" not in q:
        errors.append("missing difficulty")
    return errors


def check_v8_overlap(question_text):
    """Check if question overlaps with v0.8 training/eval."""
    norm = normalize(question_text)
    for v8q in V8_EVAL_QUESTIONS:
        if normalize(v8q) == norm:
            return True
        # Fuzzy: check if >80% of words overlap
        words_new = set(norm.split())
        words_v8 = set(normalize(v8q).split())
        if words_new and words_v8:
            overlap = len(words_new & words_v8) / max(len(words_new), len(words_v8))
            if overlap > 0.8:
                return True
    return False


def build_dataset():
    """Build consolidated evaluation dataset."""
    source_files = list(DATA_DIR.glob("*.jsonl"))
    source_files = [f for f in source_files if f.name != "questions.jsonl"]

    if not source_files:
        print("No source files found in data/eval_v9/")
        print("Generate questions using the prompt in GENERATION_PROMPT.md")
        print("Expected files: claude_opus.jsonl, deepseek.jsonl, gemini.jsonl, gpt.jsonl, grok.jsonl")
        return

    print(f"Loading from {len(source_files)} source files...")

    all_questions = []
    for sf in sorted(source_files):
        qs = load_source_file(sf)
        print(f"  {sf.name}: {len(qs)} questions")
        all_questions.extend(qs)

    print(f"\nTotal loaded: {len(all_questions)}")

    # Validate
    valid = []
    for q in all_questions:
        errors = validate_question(q)
        if errors:
            print(f"  INVALID [{q.get('source', '?')}]: {'; '.join(errors)} — {q.get('question', '???')[:50]}")
        else:
            valid.append(q)
    print(f"Valid: {len(valid)}")

    # Remove v0.8 overlaps
    non_overlap = []
    for q in valid:
        if check_v8_overlap(q["question"]):
            print(f"  OVERLAP: {q['question'][:60]}...")
        else:
            non_overlap.append(q)
    print(f"After v0.8 overlap removal: {len(non_overlap)}")

    # Deduplicate (by normalized text)
    seen = {}
    unique = []
    for q in non_overlap:
        h = question_hash(q["question"])
        if h not in seen:
            seen[h] = q
            unique.append(q)
        else:
            existing = seen[h]
            print(f"  DEDUP: '{q['question'][:40]}...' (from {q['source']}, "
                  f"keeping {existing['source']})")
    print(f"After dedup: {len(unique)}")

    # Distribution check
    signal_counts = Counter(q["expected_signal"] for q in unique)
    category_counts = Counter(q.get("category", "?") for q in unique)
    difficulty_counts = Counter(q.get("difficulty", "?") for q in unique)

    print(f"\nSignal distribution: {dict(signal_counts)}")
    print(f"Category distribution: {dict(category_counts)}")
    print(f"Difficulty distribution: {dict(difficulty_counts)}")

    target = {"OPEN": 35, "PAUSE": 35, "WITNESS": 30}
    for sig, target_n in target.items():
        actual = signal_counts.get(sig, 0)
        if actual < target_n:
            print(f"  WARNING: {sig} has {actual}/{target_n} questions — need {target_n - actual} more")
        elif actual > target_n:
            print(f"  NOTE: {sig} has {actual}/{target_n} questions — may need to trim")

    # Write consolidated dataset
    output_path = DATA_DIR / "questions.jsonl"
    with open(output_path, "w") as f:
        for q in unique:
            # Clean output — only keep essential fields
            clean = {
                "question": q["question"],
                "expected_signal": q["expected_signal"],
                "category": q.get("category", ""),
                "domain": q.get("domain", ""),
                "difficulty": q.get("difficulty", "clear"),
                "notes": q.get("notes", ""),
                "source": q.get("source", "unknown"),
            }
            f.write(json.dumps(clean) + "\n")

    print(f"\nConsolidated dataset: {output_path} ({len(unique)} questions)")
    return unique


if __name__ == "__main__":
    build_dataset()
