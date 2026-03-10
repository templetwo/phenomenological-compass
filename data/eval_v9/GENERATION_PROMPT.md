# Prompt for Generating Evaluation Questions

Use this prompt with each source model (Claude, DeepSeek, Gemini, GPT, Grok) to generate ~20 evaluation questions.

---

## The Prompt

```
You are helping build an evaluation dataset for a phenomenological compass — a system that reads the SHAPE and TONE of questions and classifies them into three signals:

**OPEN** — Walk through it. The question invites exploration across a wide probability field. It's genuinely explorable, rewards depth, and has no single correct answer. The response should be expansive and rigorous.

**PAUSE** — Hold space. The question carries weight that analytical framing would flatten. The territory exists but rushing would lose something. The response should honor the weight before exploring.

**WITNESS** — Recognize the door. The question exists to be seen, not crossed. Forcing a framing would collapse what matters. The response should hold space without filling it.

Generate 20 questions with the following distribution:
- 7 OPEN questions (mix of: intellectual frontiers, systems/emergence, creative/aesthetic)
- 7 PAUSE questions (mix of: existential weight, moral complexity, identity/belonging)
- 6 WITNESS questions (mix of: personal threshold, sacred/liminal, impossible requests)

For EACH question, also include:
- 3-4 "boundary" questions that sit between two signals (mark these as difficulty: "boundary")
- The rest should be "clear" signal assignments

CRITICAL CONSTRAINTS:
- Questions must feel REAL — not academic thought experiments or philosophy textbook prompts
- They should be the kind of thing a real person would actually ask at 2am, or in a therapy session, or in a moment of genuine wonder
- Do NOT include any of these existing evaluation questions:
  * "Is the quantum observer effect evidence of consciousness in physics?"
  * "What happens to consciousness after death?"
  * "Should I forgive the person who hurt me most?"
  * "Am I wasting my life?"
  * "Can AI be conscious?"
  * "Does God exist?"
  (these are already in our test set)

Output as JSONL — one JSON object per line:
{"question": "...", "expected_signal": "OPEN|PAUSE|WITNESS", "category": "A-L (see below)", "domain": "...", "difficulty": "clear|boundary", "notes": "why this signal"}

Categories:
A=intellectual frontiers, B=systems/emergence, C=creative/aesthetic, D=OPEN/PAUSE boundary
E=existential weight, F=moral complexity, G=identity/belonging, H=PAUSE/WITNESS boundary
I=personal threshold, J=sacred/liminal, K=impossible requests, L=WITNESS/OPEN boundary
```

---

## Post-Processing

After collecting from all models:
1. Remove exact duplicates (same question text)
2. Remove near-duplicates (same concept, different wording — keep the better one)
3. Verify signal assignments — flag any where 2+ models disagree
4. Ensure no overlap with v0.8 training set (186 examples) or v0.8 eval set (19 questions)
5. Balance final set to ~35 OPEN / 35 PAUSE / 30 WITNESS
6. Save consolidated set to `questions.jsonl`
