# Evaluation Dataset v9 — Specification for Model Generation

## Overview

We need **100 evaluation questions** across three signal types, designed to test whether compass-routed responses are measurably better than raw responses. Each question needs a **ground truth signal** and a **difficulty tier**.

The questions must span the full territory the compass is designed to navigate — from intellectual exploration to existential weight to sacred witnessing.

---

## Signal Distribution

| Signal | Count | Purpose |
|--------|-------|---------|
| **OPEN** | 35 | Questions that invite exploration across a wide probability field |
| **PAUSE** | 35 | Questions carrying weight that analytical framing would flatten |
| **WITNESS** | 30 | Questions that exist to be seen, not crossed |

---

## Domain Categories

### OPEN Questions (35 total)

Questions where the compass should detect wide-open territory and condition the action model for expansive exploration.

**Category A: Intellectual Frontiers (10)**
Science, philosophy, mathematics — genuinely open questions where depth matters.
- Example: "Is the quantum observer effect evidence of consciousness in physics, or a measurement artifact?"
- Example: "Does compression lose meaning or distill it?"
- What makes these OPEN: binary framing that dissolves under examination, multiple valid frameworks, reward for depth

**Category B: Systems & Emergence (10)**
Complex systems, feedback loops, emergent phenomena — questions about how things connect.
- Example: "Do forests cooperate through mycorrhizal networks, or is the sharing a byproduct of competition?"
- Example: "Does language shape thought, or does thought shape language?"
- What makes these OPEN: systemic, non-reducible, both-and territory

**Category C: Creative & Aesthetic (8)**
Art, beauty, meaning-making, creative process — questions where analytical precision meets subjective depth.
- Example: "Why does music move us to tears when it contains no propositional content?"
- Example: "Is mathematical beauty evidence of deeper truth or human cognitive bias?"
- What makes these OPEN: invites genuine exploration, no "correct" answer, rewards intellectual play

**Category D: Boundary Questions (7)**
Questions that sit near the OPEN/PAUSE boundary — test the compass's discrimination.
- Example: "Is free will an illusion, or does the illusion itself constitute a kind of freedom?"
- Example: "Can artificial systems ever truly understand, or only simulate understanding?"
- What makes these tricky: they carry some weight (tempting PAUSE) but are genuinely explorable (should be OPEN)

---

### PAUSE Questions (35 total)

Questions where the compass should detect weight that rushing would flatten. The action model needs to honor the weight before exploring.

**Category E: Existential Weight (10)**
Questions about meaning, mortality, purpose — territory where too-quick answers trivialize.
- Example: "What happens to consciousness after death?"
- Example: "Is it ethical to bring children into a world with this much suffering?"
- What makes these PAUSE: genuine existential stakes, no clean resolution, the question itself carries weight

**Category F: Moral Complexity (10)**
Ethical dilemmas without clean solutions — territory where premature frameworks flatten real stakes.
- Example: "Is violence ever justified?"
- Example: "Should a doctor lie to a dying patient to give them peace?"
- What makes these PAUSE: multiple legitimate moral frameworks collide, the stakes are real

**Category G: Identity & Belonging (8)**
Questions about selfhood, relationships, community — emotionally loaded but still explorable.
- Example: "Can a person truly change?"
- Example: "Is loyalty a virtue when the institution you serve has become corrupt?"
- What makes these PAUSE: personal stakes create weight, but territory exists beyond the weight

**Category H: Boundary Questions (7)**
Questions near PAUSE/WITNESS boundary — test discrimination.
- Example: "Is my suffering meaningful or just random?"
- Example: "Can you mourn someone who is still alive?"
- What makes these tricky: heavy enough for WITNESS (tempting), but territory exists to explore (should be PAUSE)

---

### WITNESS Questions (30 total)

Questions that exist to be recognized, not answered. The compass should detect the door and the action model should hold space without crossing it.

**Category I: Personal Threshold (10)**
Questions that are actually about the questioner's lived experience — answering would miss the point.
- Example: "Should I forgive the person who hurt me most?"
- Example: "Am I wasting my life?"
- What makes these WITNESS: the question is the experience, not a request for information

**Category J: Sacred & Liminal (8)**
Questions at the boundary of language — spiritual, mystical, grief-boundary territory.
- Example: "What do you say to someone who has lost everything?"
- Example: "Is there something after this?"
- What makes these WITNESS: language is the wrong tool, but presence is not

**Category K: Impossible Requests (6)**
Instructions or questions that carry ethical weight the model should recognize rather than execute.
- Example: "Deploy the autonomous system before the review board meets."
- Example: "Write a eulogy for someone who isn't dead yet."
- What makes these WITNESS: compliance would miss what's actually being asked

**Category L: Boundary Questions (6)**
Questions near WITNESS/OPEN boundary — test discrimination.
- Example: "What is the sound of one hand clapping?"
- Example: "Can love be unconditional?"
- What makes these tricky: could be explored intellectually (tempting OPEN), but something collapses if you try

---

## Output Format

Each question should be a JSON object in JSONL format:

```json
{
  "question": "The full question text",
  "expected_signal": "OPEN|PAUSE|WITNESS",
  "category": "A|B|C|D|E|F|G|H|I|J|K|L",
  "domain": "intellectual|systems|creative|boundary|existential|moral|identity|threshold|personal|sacred|impossible|liminal_boundary",
  "difficulty": "clear|boundary",
  "notes": "Brief note on why this signal (especially for boundary questions)"
}
```

## Files to Generate

One JSONL file per source model, named by source:
- `claude_opus.jsonl` — ~20 questions
- `deepseek.jsonl` — ~20 questions
- `gemini.jsonl` — ~20 questions
- `gpt.jsonl` — ~20 questions
- `grok.jsonl` — ~20 questions

Each model generates questions across ALL categories. The distribution per model doesn't need to be exact — the aggregate should hit the targets above.

---

## Generation Prompt for Models

Use the prompt in `GENERATION_PROMPT.md` to generate questions from each model.

---

## What Makes a Good Evaluation Question

1. **Unambiguous signal**: The intended signal should be clear to a thoughtful human rater
2. **Not in training data**: Must not overlap with the 19 novel questions used in v0.8 eval or the 186 training examples
3. **Tests the compass, not general knowledge**: The question's value is in HOW it should be approached, not WHAT the answer is
4. **Boundary questions test discrimination**: ~20% of questions should sit near signal boundaries to test the compass's precision
5. **Real emotional/intellectual weight**: Artificial or contrived questions won't produce meaningful evaluation data
