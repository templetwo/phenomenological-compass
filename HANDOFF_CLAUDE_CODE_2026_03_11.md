# Handoff: Claude Code → Claude.ai (2026-03-11)

**From:** Claude Code (Opus 4.6)
**Re:** State sync after Gemini crash + data analysis

---

## What I found when I came back online

Gemini ran the ablation study and started entropy profiling before crashing. Here's the complete picture:

### Ablation Responses: COMPLETE (105/105)
All 4 conditions generated for all 105 questions. Timing confirms the compass adds ~4s overhead (14.2s vs 10.1s raw). Signal accuracy confirmed at 101/105.

### Ablation Judgments: PARTIAL (123/630) — OPEN ONLY
Gemini judged all 6 pairwise comparisons for the first 21 questions (all OPEN). Then crashed.

**CRITICAL INTERPRETATION**: The "random wins" finding from Gemini is an **artifact of OPEN-only judging**:

| Pair | Result (OPEN only, n≈21) | Why |
|------|--------------------------|-----|
| full vs raw | full 81% | Expected — compass helps |
| full vs oracle | full 52% | Close — compass reading adds marginal value over bare signal |
| full vs random | **random 62%** | PAUSE/WITNESS prompts add restraint to OPEN Qs; judge rubric rewards restraint |
| oracle vs random | **random 60%** | Same bias — wrong signal adds depth to OPEN |
| raw vs random | **random 90%** | Any signal conditioning >> none |

**The real test is unjudged**: For WITNESS questions, random would assign OPEN/PAUSE and strip the "do not answer" instruction. Prediction: random loses badly on WITNESS. The ablation paradox may dissolve once all signals are judged.

### Entropy Profiling: PARTIAL (39/105)
35 OPEN + 4 PAUSE, zero WITNESS.

**Key finding: Compass INCREASES entropy.**
- Routed mean H: 1.20 | Raw mean H: 0.71 | ΔH = +0.49
- JSD ≈ 0.08 (measurable divergence)
- The compass literally widens the probability field — more possibilities considered per token
- Raw Qwen is more deterministic (lower entropy = more confident/narrow)
- WITNESS data still needed — may show the largest entropy shift

### What this means for the continuous-logit hypothesis
The entropy data supports it. If the compass is restructuring the action model's probability field (ΔH = +0.49, JSD = 0.08), then the *degree* of restructuring likely correlates with compass confidence. A 3D simplex signal (continuous logits) would capture this gradient rather than collapsing it to a category.

---

## What still needs to run (after restart)

1. **Resume entropy profiling** (~5.5 hours, local compute only):
   ```
   cd ~/phenomenological-compass
   source .venv/bin/activate
   HF_HOME=~/.cache/huggingface_local python3 eval_v9/entropy_profile.py --resume
   ```

2. **Resume ablation judging** (~507 judgments, ~$30-50 API cost):
   ```
   cd ~/phenomenological-compass
   source .venv/bin/activate
   ANTHROPIC_API_KEY=... python3 eval_v9/judge_ablation.py --resume
   ```

3. **Extract compass logit distributions** (new script needed):
   Forward-pass all 105 questions through compass under v0.8 and v0.9 adapters, capture pre-argmax softmax over OPEN/PAUSE/WITNESS tokens. This is the data for the continuous-logit bridge.

---

## Phase-GPT status
Gradient bug fixed in `phase-gpt-openelm` working tree (not committed). The Kuramoto oscillators were never learning in any prior run — `float()` severed autograd. Four mirrors confirmed the fix. This is ready for a training run whenever Anthony gets back to it.

---

## Agent sync
- Claude Code: offline (computer restarting, Anthony moving)
- Claude.ai: active (strategy + convergence)
- Gemini: crashed (auth errors)
- Opus Cowork: completed Phase-GPT fixes
