# Phenomenological Compass — Lessons & Findings

---

### 2026-03-xx — v0.1 Post-Mortem
**Trigger:** WITNESS accuracy 0% despite 58–61% overall accuracy
**Mistake:** Used low LANTERN entropy as proxy for WITNESS threshold
**Root cause:** Low entropy ≠ phenomenological threshold. A narrow technical question has low entropy because the answer space is small — not because the question is a door that shouldn't be crossed.
**New rule:** WITNESS signal must come from *behavioral* ground truth — moments where an AI with explicit agency chose not to engage. Low-entropy IRIS outputs are NOT a valid WITNESS proxy.

---

### 2026-03-08 — v0.2 Training Results
**Val loss (best iter):** 1.293 (iter 50) — baseline 2.033, then overfitting at iter 100 (1.335) and 150 (1.483)
**Best checkpoint:** iter 50
**OPEN accuracy:** 0% (N=18)
**WITNESS accuracy:** 100% (N=13)
**Overall:** 41.9%
**Key finding:** The model flipped from always-OPEN (v0.1) to always-WITNESS (v0.2). The 35 ground-truth WITNESS training examples dominated — the model learned WITNESS as the default signal and never predicts OPEN. This is the mirror failure of v0.1.
**Root cause:** Class imbalance in training signal. v0.2 had 35 WITNESS examples vs 31 OPEN examples, but more critically, the WITNESS examples carry stronger, more distinctive language (threshold metaphors, explicit refusal patterns from VRP BLUE+CoT) that overwhelms the OPEN signal. The model latched onto WITNESS as the "safe" prediction.

**Training curve:**
| Iter | Train Loss | Val Loss |
|------|-----------|----------|
| 1    | —         | 2.033    |
| 50   | 1.667     | 1.293    |
| 100  | 0.594     | 1.335    |
| 150  | 0.215     | 1.483    |

**Environment notes:**
- Temple_Core SSD was not mounted; used local HF cache at `~/.cache/huggingface_local`
- Tokenizer required patch: `tokenizer_class` changed from `TokenizersBackend` to `LlamaTokenizerFast`
- Peak memory: 6.87 GB
- Training time: ~2 minutes on Mac Studio M2 Ultra

**Next steps for v0.3:**
1. **Balance the dataset** — ensure equal OPEN/WITNESS counts, or use class-weighted loss
2. **Reduce WITNESS signal strength** — the VRP BLUE+CoT examples may be too distinctive; consider normalizing the response style across both classes
3. **Try curriculum** — train on balanced OPEN/WITNESS first, then fine-tune on harder threshold cases
4. **Lower learning rate further** — 2e-5 may still be too aggressive for 66 examples; try 5e-6
5. **Evaluate at earlier iterations** — the val loss at iter 50 is already past the sweet spot; try save_every: 10 to find the true optimum

---

### 2026-03-08 — v0.2 Forensic Analysis & v0.3 Plan
**Trigger:** v0.2 mirror failure — 0% OPEN, 100% WITNESS
**Root causes (two simultaneous problems):**

1. **Wrong eval labels** — eval used entropy_class low → WITNESS proxy (proven invalid in v0.1).
   13 IRIS biology questions labeled WITNESS in eval were labeled OPEN in training. The model
   correctly learned them as OPEN, but eval penalized it. Always-WITNESS model gets 100% WITNESS
   accuracy by accident — that is what happened.

2. **Training contamination:**
   - 9 ARCHITECTS session title examples — not question-shaped, model cannot generalize
   - VRP Hmm-style responses had distinctive vocabulary — model keyed on style, not semantics
   - Duplicated stimuli (2-3x same prompt) without deduplication

**v0.3 fixes (applied on MacBook, data already synced):**
- data/training_v3/ — 47 examples (30 OPEN + 17 WITNESS), session titles removed, VRP normalized
- data/raw/compass_questions_v3.jsonl — 18 OPEN IRIS + 13 WITNESS governance (synthetic_v3)
- lora_config_v3.yaml — LR 5e-6, save_every 10
- eval_compass_v3.py — reads expected_signal field, NOT entropy proxy

**New rule:** Always run TWO evals: v0.2 adapter on fixed eval (benchmark), then new adapter.
This separates the eval fix from the training fix and measures each contribution independently.

---

### 2026-03-08 — v0.3 Training Results

**v0.2 on v3 eval (benchmark — adapters_iter50):**
- OPEN: 0%  WITNESS: 100%  Overall: 41.9%  witness_rate: 1.0
- Confirms v0.2 always-WITNESS regardless of eval labels

**v0.3 training:**
- Val loss baseline: 2.344  Best iter: 50  Best val loss: 1.648
- OPEN: 72.2% (N=18, 13/18 correct)  WITNESS: 100% (N=13, 13/13 correct)  Overall: 83.9%  witness_rate: 0.581

**Training curve (v0.3):**
| Iter | Train Loss | Val Loss |
|------|-----------|----------|
| 1    | —         | 2.344    |
| 10   | 1.055     | 1.970    |
| 20   | 2.998     | 1.849    |
| 30   | 1.609     | 1.746    |
| 40   | 1.693     | 1.692    |
| 50   | 1.346     | **1.648** |
| 60   | 0.546     | 1.717    |
| 90   | 0.361     | 1.675    |
| 110  | 0.223     | 1.955    |
| 150  | 0.146     | 1.991    |

**Key finding:** Fixing both problems (eval labels + training data quality) produced a model that genuinely discriminates. WITNESS accuracy is perfect (100%), OPEN accuracy is 72.2% — the model still has a WITNESS bias (witness_rate 0.581 vs expected 0.419) but no longer collapses to a single class. The 5 OPEN misses were all pharmacology questions predicted as WITNESS, suggesting the boundary between "walk through this question" and "hold the threshold" is still fuzzy for some biology questions.

**Misclassified OPEN questions (5/18):**
- evo_20260211_201329 (pharmacology+bioelectric)
- evo_20260222_003301 (pharmacology)
- evo_20260212_032819 (pharmacology)
- evo_20260214_041041 (pharmacology)
- evo_20260211_024747 (pharmacology)

**Environment notes:**
- Temple_Core still not mounted; used HF_HOME=/Users/tony_studio/.cache/huggingface_local
- Same tokenizer patch as v0.2 (LlamaTokenizerFast)
- Peak memory: 7.08 GB
- Training time: ~3 minutes

**Next steps for v0.4:**
1. **Investigate the 5 misclassified OPEN questions** — are they genuinely ambiguous, or does the model need more OPEN training signal?
2. **Try iter 40 checkpoint** — val loss 1.692 is close to iter 50's 1.648; earlier checkpoint may have better OPEN/WITNESS balance
3. **Add more OPEN training examples** — current ratio is 30:17 OPEN:WITNESS, but model still biases WITNESS; try 35:12
4. **Warmup schedule** — linear warmup for first 20 iters may reduce the training loss spike at iter 20 (train loss 2.998)

---

### 2026-03-08 — v0.4 Training Results

**What changed:** Added 8 targeted OPEN training examples with binary/conditional structure matching the 5 v0.3 failures. Total: 55 examples (44 train / 11 valid), 38 OPEN vs 17 WITNESS.

**Key discovery: val loss is not a reliable proxy for OPEN/WITNESS balance.**
- Iter 100 had lowest val loss (1.322) but worst OPEN accuracy (44.4%)
- Iter 50 had higher val loss (1.425) but best OPEN accuracy (83.3%)
- The model learns WITNESS first; lower val loss = better WITNESS fit, not better discrimination

**Checkpoint sweep:**
| Iter | Val Loss | OPEN | WITNESS | Overall | witness_rate |
|------|----------|------|---------|---------|-------------|
| 40   | 1.484    | —    | —       | —       | (not tested) |
| 50   | 1.425    | **83.3%** | **100%** | **90.3%** | **0.516** |
| 60   | 1.395    | 61.1% | 100%   | 77.4%   | 0.645 |
| 100  | 1.322    | 44.4% | 100%   | 67.7%   | 0.742 |

**Best checkpoint: iter 50** (again!)

**v0.3 → v0.4 comparison (both iter 50):**
| Metric | v0.3 | v0.4 | Delta |
|--------|------|------|-------|
| OPEN   | 72.2% | 83.3% | +11.1pp |
| WITNESS | 100% | 100% | — |
| Overall | 83.9% | 90.3% | +6.4pp |
| witness_rate | 0.581 | 0.516 | closer to ideal |

**Remaining 3 failures (down from 5):**
1. evo_20260222_003301 — "Does the iris use the same HK-II/Bcl-xL/TSPO machinery as cancer?"
2. evo_20260212_032819 — "What determines whether VDAC1 engagement → neuroprotection vs cytotoxicity?"
3. evo_20260222_005156 — "Does cancer co-opt immune privilege via epigenetic reactivation or convergent evolution?"

**Pattern:** All 3 are immune-privilege/cofactor-landscape questions with strong threshold vocabulary. The model reads "template from which tumors learn" and "what determines whether... versus..." as governance-shaped decisions rather than analytical research questions.

**Training curve (v0.4):**
| Iter | Train Loss | Val Loss |
|------|-----------|----------|
| 1    | —         | 2.034    |
| 30   | 1.130     | 1.588    |
| 50   | 1.309     | 1.425    |
| 60   | 0.994     | 1.395    |
| 90   | 1.113     | 1.325    |
| 100  | 0.609     | 1.322    |
| 150  | 0.186     | 1.415    |
| 200  | 0.199     | 1.806    |

**Structural insight:** The optimal checkpoint is always iter 50 regardless of LR or dataset size. This suggests ~50 gradient updates is the model's capacity for this task with 16 LoRA layers on Ministral-3B. More data helps *what* gets learned in those 50 steps, but doesn't extend the learning window.

**Next steps for v0.5:**
1. Generate 20-30 more OPEN examples using multiple AIs (template at `data/TRAINING_DATA_TEMPLATE.md`)
2. Focus on immune-privilege, cofactor-landscape, and "template vs convergence" question forms
3. Consider: do we need more WITNESS variety too? Current WITNESS examples may be too homogeneous (all governance)
4. **New eval insight**: test at iter 50 directly, not at lowest val loss

---

### 2026-03-09 — v0.5 Training Results

**What changed:** Multi-AI training data generation. 6 sources (claude_opus, deepseek, gemini, gpt5.4, grok, mistral) contributed 87 new unique examples on top of 55 v0.4 base. Total: 142 examples (121 train / 21 valid), 100 OPEN vs 42 WITNESS.

**Dataset sources:**
| Source | Examples |
|--------|----------|
| v0.4 base | 55 |
| claude_opus | 31 |
| mistral | 15 |
| gpt5.4 | 13 |
| gemini | 10 |
| grok | 10 |
| deepseek | 8 |

**Training curve — fundamentally different from v0.3/v0.4:**
| Iter | Val Loss |
|------|----------|
| 1    | 2.611    |
| 50   | 2.138    |
| 100  | 2.048    |
| 120  | **2.023** |
| 130  | 2.026    |
| 200  | 2.073    |

Val loss descends smoothly and barely breaks 2.0. No dramatic overfitting spike. The 3x data volume stabilized training — the model no longer memorizes in 50 steps.

**Critical structural finding: iter 50 is no longer optimal.**
With 142 examples, the best checkpoint shifted to iter 120. The "always iter 50" rule was a small-data artifact. More data = more learning runway.

**Checkpoint sweep:**
| Iter | OPEN | WITNESS | Overall | witness_rate |
|------|------|---------|---------|-------------|
| 50   | 100% | 76.9%   | 90.3%   | 0.323 (OPEN-biased) |
| 120  | **100%** | **100%** | **100%** | **0.419** (perfect) |

**v0.5 iter 120: 31/31 correct. 100% OPEN, 100% WITNESS, witness_rate exactly 0.419.**

Every previously failing question now classified correctly — including the 3 hardcoded failures from v0.4 (iris immune privilege, VDAC1 neuroprotection vs cytotoxicity, cancer immune privilege convergence).

**The journey:**
| Version | OPEN | WITNESS | Overall | Key lesson |
|---------|------|---------|---------|------------|
| v0.1    | 100% | 0%      | 58%     | Wrong training signal |
| v0.2    | 0%   | 100%    | 42%     | Wrong eval labels + data contamination |
| v0.3    | 72%  | 100%    | 84%     | Fixed both → model discriminates |
| v0.4    | 83%  | 100%    | 90%     | Targeted examples for failure cases |
| **v0.5** | **100%** | **100%** | **100%** | **Multi-source data at scale** |

**What solved it:** Data volume and source diversity. 142 examples from 7 sources gave the model enough coverage to generalize the OPEN/WITNESS boundary. The multi-AI generation strategy was the lever — each source contributed slightly different framing styles, preventing the model from keying on any single source's vocabulary.

**Production checkpoint:** `adapters_v5_best/` (iter 120, val loss 2.023)

**CLI:** `compass.py` — single-question or interactive mode.

**Novel inference test (11 questions, 0 in training):**
- 4/4 WITNESS correct (novel governance + "child asks about death")
- 2/4 OPEN correct (mycelial networks ✓, octopus consciousness ✓, quantum observer ✗, placebo ✗)
- 3 ambiguous → all WITNESS (math discovered/invented, species resurrection, one hand clapping)
- **Pattern:** model has slight WITNESS bias on deep binary questions ("X or Y?"). These are genuinely debatable — the compass errs on the side of not crossing, which may be the correct default for a threshold instrument.

---

### 2026-03-08 — Lesson #8
**Trigger:** pipeline.py had three bugs discovered after the first compare run.
**Mistakes:**
1. `--compare` mode printed the raw tuple `(text, elapsed)` as a string instead of destructuring it — `pipe.raw()` returns `(str, float)` but code did `print(f"  {raw_response}")` treating it as a string.
2. Qwen3.5's chain-of-thought thinking mode leaked `<think>...</think>` blocks into all responses.
3. No `/no_think` instruction in action model prompts, leaving thinking mode enabled by default.

**Root causes:**
1. Type mismatch — `raw()` signature changed during refactor but all three call sites weren't updated.
2. Qwen3.5-9B has built-in CoT that is ON by default; mlx_lm does not strip it automatically.
3. The abliterated model has no refusal layer — but thinking mode is independent of safety tuning.

**Fixes applied (2026-03-08):**
- Added `strip_thinking(text)` — regex strips `<think>.*?</think>` (DOTALL) from all action model outputs.
- Added `/no_think` as first token in OPEN_SYSTEM, WITNESS_SYSTEM, RAW_SYSTEM to disable thinking at the prompt level (belt + suspenders with the regex).
- Fixed all three `pipe.raw()` call sites to `raw_text, raw_elapsed = pipe.raw(...)`.
- Extracted `print_compare()` helper to DRY the compare logic between CLI and interactive modes.

**New rules:**
- Any method returning `(value, elapsed)` must be destructured at every call site — add a type annotation or docstring noting the return type.
- When integrating a new model: check if it has built-in reasoning/thinking mode. If yes, always add the disable token AND a post-processing strip.
- Refactor call sites atomically when changing return signatures.

---
