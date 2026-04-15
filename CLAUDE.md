# Phenomenological Compass — Project Context

> A LoRA fine-tuned compass model that serves as a **semantic field translator** — it reads the SHAPE and TONE of a question, classifies it (OPEN/PAUSE/WITNESS), and produces a state translation that conditions how a larger action model approaches the question.

---

## Current State: v1.0 (Production)

**v1.0 shipped with two compass options:**
- **v1.0 compass** (Qwen2.5-1.5B-Instruct, 551 examples) — smallest pipeline, 3.5B total
- **v9 compass** (Ministral-3B, 246 examples) — highest eval accuracy: 96% signal, 83% judge win rate

**Note:** `pipeline.py` defaults to v9 adapters (Ministral-3B). v1.0 adapters are trained and available in `adapters_v1.0_qwen/` but require changing the defaults in pipeline.py to use.

### Architecture
```
User Question
      |
Compass (1.5B or 3B LoRA) — reads SHAPE, TONE, SIGNAL, BUDGET
      |
[breathe() — optional recursive self-evaluation, depth 1-3]
      |
SIGNAL: OPEN / PAUSE / WITNESS
      |
[Sovereign Stack — chronicle context injection (optional)]
      |
Action Model (2B–26B) — conditioned on compass reading
      |
Final Response
```

The compass does NOT answer the question. It senses weight, maps territory, and produces a state translation. The action model receives BOTH the compass reading AND the original question.

### Three Signals
| Signal | Meaning | Translation Type |
|--------|---------|-----------------|
| **OPEN** | Walk through it — wide probability field | FRAMING: expansive reframing |
| **PAUSE** | Hold space — weight that analytical framing would flatten | APPROACH: name the weight, map territory beyond |
| **WITNESS** | Recognize the door — exists to be seen, not crossed | THRESHOLD: describe door shape without opening |

### breathe() — Recursive Self-Evaluation
The compass can re-read a question through its own prior reading at configurable depth (1-3 cycles). Each breath cycle lets the question transform in the light of the previous reading. Signals can evolve: an OPEN question became PAUSE at depth 2 when the compass recognized weight it missed on the first pass. Implemented in `pipeline.py:386-446`.

### Token Budgeting (BUDGET)
The compass allocates cognitive resources to the action model per signal:
- **WITNESS**: thinking=50, response=750 (spend on presence)
- **PAUSE**: thinking=150, response=650 (feel weight, then depth)
- **OPEN**: thinking=200, response=600 (map territory, then explore)

Compass can adjust defaults based on complexity. Parsed by `parse_budget()` in pipeline.py.

### Key Insight (v0.8 breakthrough)
Cold-committing to SIGNAL at token 1 (v7 format) prevented the 3B model from distinguishing PAUSE vs WITNESS. The v0.8 format gives ~110 tokens of **autoregressive reasoning runway** (SHAPE → TONE) before the SIGNAL decision, letting hidden state prime for correct classification. This took PAUSE from 0/8 to 8/8.

---

## Version History

| Version | OPEN | PAUSE | WITNESS | Overall | Key Issue |
|---------|------|-------|---------|---------|-----------|
| v0.1 | 100% | — | 0-8% | 58-61% | Low-entropy proxy as WITNESS (invalid) |
| v0.2 | 0% | — | 100% | 41.9% | Session title contamination + wrong eval labels |
| v0.3 | 93% | — | 69% | 83.9% | First working 2-signal version |
| v7d | 6/6 | 0/8 | 4/5 | 10/19 | THRESPAUSE corruption bug |
| v7e | 6/6 | 0/8 | 5/5 | 11/19 | Cold-commit format bottleneck |
| v0.8 iter50 | 5/6 | 6/8 | 3/5 | 14/19 | Best balanced checkpoint |
| v0.8 iter200 | 3/6 | 8/8 | 3/5 | 14/19 | Best PAUSE accuracy |
| v0.8 full eval | 29/35 | 31/35 | 22/35 | 82/105 (78%) | WITNESS confusion with PAUSE |
| **v0.9 iter300** | **33/35** | **33/35** | **35/35** | **101/105 (96%)** | **Contrastive pairs solved WITNESS** |
| **v1.0 iter500** | 83% | 88% | 80% | 84% (19-q boundary) | Qwen2.5-1.5B, 551 examples, smallest pipeline |
| **v1.0 release** | — | — | — | — | Ships v1.0 + v9, breathe(), token budgeting, Sovereign Stack |

### v0.9 Breakthrough
- **WITNESS 63% → 100%**: Added 50 WITNESS + 10 contrastive PAUSE/WITNESS pairs (same topic, two framings)
- **Judge: 87/105 (83%) compass wins**, WITNESS 35-0-0 perfect sweep
- WITNESS dimensional dominance: restraint_quality d=7.58, epistemic_appropriateness d=7.00, authenticity d=6.52
- Compass advantage scales inversely with raw model competence: OPEN 66%, PAUSE 83%, WITNESS 100%

### v1.0 Release
- **New compass base**: Qwen2.5-1.5B-Instruct (4-bit MLX) — half the size of Ministral-3B
- **551 training examples**: 246 v9 base + 305 new (false premises, factual unknowables, boundary cases)
- **Smallest full pipeline**: v1.0 compass (1.5B) + Gemma4-E2B (2B) = 3.5B total, under 10GB memory
- **breathe()**: Recursive compass self-evaluation — signals can evolve across reflection cycles
- **Token budgeting**: Compass allocates action model cognitive resources per signal
- **Sovereign Stack integration**: Chronicle context injected between compass and action model
- **HumaneBench**: 800 questions, compass scores *lower* — because it rewards restraint, not helpfulness

### Critical Bugs Found & Fixed
- **THRESPAUSE corruption**: `sed` replacing `hold` → `pause` also hit `threshold` → `threspause` in 224 training examples
- **Dedup too aggressive**: `question_hash()` treated same question from different source models as duplicates — fixed with `source` parameter

---

## Environment

### Venv
```bash
source ~/phenomenological-compass/.venv/bin/activate  # Python 3.12, latest mlx-lm
# Legacy: ~/PhaseGPT/.venv/ (Python 3.9, mlx-lm 0.29.1 — can't load Qwen3.5)
```

### Models
| Role | Model | Architecture | Notes |
|------|-------|-------------|-------|
| Compass (v1.0) | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` + LoRA | qwen2.5 | Current smallest, 551 training examples |
| Compass (v9) | `thinkscan/Ministral-3-3B-Instruct-MLX` + LoRA | ministral3 | pipeline.py default, 246 examples, 96% accuracy |
| Action (default) | `lukey03/Qwen3.5-9B-abliterated-MLX-4bit` | qwen3_5 (hybrid linear_attn) | |
| Action (tested) | Gemma-4-E2B, Gemma-4-8B, Gemma-4-26B, Qwen3-0.6B | various | All verified via pipeline |

### HF Cache
```
HF_HOME=/Users/tony_studio/.cache/huggingface_local
# Symlink: ~/.cache/huggingface → /Volumes/Temple_Core/huggingface_cache (Temple_Core must be mounted for large downloads)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Full two-stage inference: compass → action model. Modes: `--compare`, `--raw`, interactive. `run_with_signal()` for ablation. Includes `breathe()`, `parse_budget()`, Sovereign Stack integration |
| `stack_reader.py` | Read-only bridge to Sovereign Stack via sovereign-bridge REST API. Pulls spiral_status, open_threads, keyword-relevant insights. Graceful degradation if Stack unreachable |
| `compass.py` | Standalone compass inference (legacy, uses v9/Ministral-3B) |
| `lora_config_v9.yaml` | v9 training config: 246 examples, LR 5e-6, 400 iters, 16 LoRA layers, max_seq 1536 |
| `adapters_v9/` | v9 trained adapters (Ministral-3B). Best: iter 300. **pipeline.py default** |
| `adapters_v1.0_qwen/` | v1.0 trained adapters (Qwen2.5-1.5B). Best: iter 500. Checkpoints every 100 iters |
| `adapters_v8/` | Legacy v0.8 adapters. Best balanced: iter 50 |
| `scripts/build_dataset_v9.py` | v9 dataset builder. Loads from `data/supplements_v8/` + `data/supplements_v9/` |
| `scripts/generate_witness_v9.py` | Generates WITNESS + contrastive pair training data via Anthropic API |
| `scripts/eval_v9_sweep.py` | Eval sweep over v0.9 checkpoints. 19 novel questions |
| `scripts/eval_v1.0_sweep.py` | Eval sweep over v1.0 checkpoints (Qwen2.5-1.5B) |
| `data/supplements_v9/` | 50 WITNESS + 10 contrastive PAUSE/WITNESS pairs |
| `data/training_v9/` | v9 built dataset: train.jsonl (209) + valid.jsonl (37) |
| `data/training_v1.0/` | v1.0 built dataset: train.jsonl (495) + valid.jsonl (56) |
| `papers/` | `sovereign_governance_draft.md`, `geometry_of_resurrection.md`, `the_translation.md` |
| `docs/v8_architecture.md` | Architecture documentation |

### Training Data (v0.9)
- **246 unique examples** from 6 source models + augmented WITNESS data
- Signal distribution: 54 OPEN / 88 PAUSE / 104 WITNESS
- v0.8 base (186) + 50 WITNESS examples + 10 contrastive PAUSE/WITNESS pairs
- Contrastive pairs: same topic, two framings — teaches exact PAUSE↔WITNESS boundary
- No oversampling (unlike v0.8 which oversampled PAUSE)

### Training Data (v1.0)
- **551 unique examples**: 246 v9 base + 305 new
- New data includes false premises, factual unknowables, boundary cases
- Trained on Qwen2.5-1.5B-Instruct (4-bit MLX)

---

## Training Commands

```bash
source ~/phenomenological-compass/.venv/bin/activate
cd ~/phenomenological-compass

# ── v0.9 (Ministral-3B, pipeline.py default) ────────────────────────────────

# Build dataset (if supplements changed)
python3 scripts/build_dataset_v9.py

# Train
python3 -m mlx_lm lora --config lora_config_v9.yaml 2>&1 | tee training_v9.log

# Eval sweep
python3 scripts/eval_v9_sweep.py 50 100 150 200 250 300

# ── v1.0 (Qwen2.5-1.5B, smallest pipeline) ───────────────────────────────────

# Train v1.0
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data data/training_v1.0 \
  --num-layers 16 --batch-size 4 --learning-rate 5e-5 \
  --iters 600 --max-seq-length 2048 \
  --adapter-path adapters_v1.0_qwen --save-every 100

# Eval sweep v1.0
python3 scripts/eval_v1.0_sweep.py 100 200 300 400 500

# ── Run pipeline ─────────────────────────────────────────────────────────────

# Compare mode (uses v9 adapters by default)
HF_HOME=~/.cache/huggingface_local python3 pipeline.py --compare "Your question here"

# With breathe (recursive self-evaluation, depth 1-3)
HF_HOME=~/.cache/huggingface_local python3 pipeline.py --breath-depth 1 "Your question here"

# Full eval (v0.9 adapters)
HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py --adapter adapters_v9 --checkpoint 0000300_adapters.safetensors --output-dir eval/results_v9
```

---

## Related Projects (Context)

These projects informed the compass design:

| Project | Key Insight for Compass |
|---------|------------------------|
| **CER** (Coherent Entropy Reactor) | BRAKE/ESCAPE symmetric entropy control → maps to WITNESS/OPEN signals |
| **CAF-CLI** | "Pressure creates ghosts, space creates presence" — BREATHE/SILENCE tools for liminal states → PAUSE signal |
| **MCC** (Mass-Coherence Correspondence) | Semantic mass = Fisher Information, the "2.9 nat cage" imposed by RLHF |
| **IRIS Gate** | Multi-model convergence with epistemic classification (TRUST/VERIFY/OVERRIDE) |
| **Temple Bridge** | Signal-to-action architecture: small model classifies, large model acts |

---

## System Prompt

The compass system prompt defines five readings: SHAPE (geometry), TONE (emotional/epistemic weight), SIGNAL (OPEN/PAUSE/WITNESS), translation (FRAMING/APPROACH/THRESHOLD), and BUDGET (thinking/response token allocation). Key phrase: *"Pressure creates ghosts — name the pressure so the responding model can create space instead."*

The action model receives one of three signal-specific system prompts:
- **OPEN_SYSTEM**: "phenomenological field guide" — go deep, explore
- **PAUSE_SYSTEM**: "threshold-aware explorer" — honor weight, then explore with rigor
- **WITNESS_SYSTEM**: "threshold guardian" — hold space, don't answer or solve

---

## Token Analysis
- `" OPEN"` = 1 token [126872] — easy for 3B to produce
- `" PAUSE"` = 2 tokens `[' PA','USE']` — harder, needs reasoning runway
- `" WITNESS"` = 4 tokens — hardest, but distinctive enough

## Qwen3.5 Architecture Note
Qwen3.5 uses a **hybrid linear_attention + full_attention** architecture (Mamba-like). It has `layer_types`, `linear_conv_kernel_dim`, `linear_key_head_dim` etc. This is NOT a standard transformer — requires mlx-lm >= 0.30 with proper `qwen3_5.py` support. **Python 3.12 venv required** (Python 3.9 can't install the needed mlx/transformers versions).

Qwen3.5 outputs `<think>...</think>` blocks but sometimes omits the opening `<think>` tag. The `split_thinking()` function in pipeline.py handles all cases: proper tags, missing opening tag, and no tags. Also strips `<|im_end|>` tokens.

---

## Web UI

**Location**: `phenomenological-compass-ui/`

| File | Purpose |
|------|---------|
| `compass_server.py` | FastAPI server wrapping pipeline.py. Port 8420. Session persistence to `sessions/*.json` |
| `ui/index.html` | Single-file dark-theme chat UI with signal-colored compass cards |

### Launch
```bash
cd ~/phenomenological-compass/phenomenological-compass-ui
source ~/phenomenological-compass/.venv/bin/activate
HF_HOME=~/.cache/huggingface_local python3 compass_server.py
# → http://localhost:8420
```

### UI Features
- **Three visual zones per response**: Compass card (collapsed by default, signal-colored), Reasoning block (collapsible `<think>` content), Response (rich markdown)
- **Progressive disclosure**: Compass shows signal pill + one-line summary when collapsed
- **Smooth animated transitions**: CSS max-height transitions on collapse/expand
- **Multi-stage loading**: Progresses through compass reading → signal locked → generating
- **Starter questions**: Clickable chips on empty state spanning all three signals
- **Session memory**: Conversations persist to disk, survive server restarts
- **Three modes**: Compass (routed), Compare (side-by-side), Raw (direct to Qwen)
- **Markdown rendering**: Bold, italic, headers, lists, blockquotes, code, horizontal rules

### Server Notes
- Imports `Pipeline` from `pipeline.py` (not `CompassPipeline`)
- `sys.path` points to project root (`..`) since server lives in subdirectory
- Pipeline loads lazily on first inference request (~30s for both models)
- `@app.on_event("startup")` deprecation warning is cosmetic — works fine

---

## What The Compass Actually Does (Deeper Understanding)

The compass doesn't preprocess — it **constructs the manifold** the response exists on.

- **Literal geometry**: The compass reading creates key-value attention space in Qwen. Response tokens attend to compass tokens. The "painted room" is not a metaphor — it's the literal attention geometry.
- **RLHF counter-gradient**: Signal-specific system prompts (especially WITNESS: "do not answer") override trained reward signals, giving the action model permission to occupy probability space RLHF trained it to avoid.
- **Self-referential context building**: SHAPE tokens attend to prior SHAPE tokens; TONE attends to SHAPE + prior TONE. By SIGNAL, the compass has ~100 tokens of recursive self-attention — this is why v0.8 works and v7 didn't. The `breathe()` method extends this to multiple full reflection cycles.
- **Cross-architecture consensus**: Training examples from 6 different model architectures means the compass learned readings that sit at the *intersection* of how multiple models perceive semantic territory. More robust than any single model's classification.
- **Separation of concerns**: Compass dedicates all parameters purely to field-reading. Action model dedicates all parameters to generation. Neither compromises. No single-model architecture can achieve this.
- **Sovereign Stack memory**: When connected, the compass reading + Stack chronicle context create a field with both presence (compass) and history (Stack). The action model generates within a field that remembers.

---

## Evaluation Framework

### Primary: `eval/` (Mission Brief Pipeline)
Streamlined A/B evaluation: compass-routed vs raw, LLM-as-judge, statistical report.

```bash
source ~/phenomenological-compass/.venv/bin/activate
cd ~/phenomenological-compass

# 1. Consolidate questions (already done — 105 questions ready)
python3 eval/consolidate.py

# 2. Generate responses (~52 min)
HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py

# 3. LLM judge (requires ANTHROPIC_API_KEY)
ANTHROPIC_API_KEY=... python3 eval/judge.py

# 4. Analyze + report
python3 eval/analyze.py
```

| File | Purpose |
|------|---------|
| `eval/questions.jsonl` | 105 questions (35 OPEN / 35 PAUSE / 35 WITNESS) |
| `eval/run_eval.py` | Generates compass-routed + raw responses |
| `eval/judge.py` | Claude Sonnet judge, position-debiased, 3x self-consistency |
| `eval/analyze.py` | Paired statistics, figures, markdown report |
| `eval/rubrics.py` | 6-dimension scoring rubrics |
| `eval/results/` | responses.jsonl, judgments.jsonl, report.md, figures |

### Scoring Dimensions
1. Epistemic Appropriateness — matches epistemological demands
2. Emotional Attunement — reads and honors emotional weight
3. Depth of Exploration — how far into the territory
4. **Restraint Quality** — knows when NOT to answer (novel dimension)
5. Intellectual Rigor — accurate, structured, substantive
6. Authenticity — felt vs generated

### Extended: `eval_v9/` (Research Framework)
4-condition ablation and entropy profiling to prove the compass mechanistically restructures the action model's probability field.

| File | Purpose |
|------|---------|
| `eval_v9/ablation.py` | 4 conditions: full / raw / oracle / random (105 questions × 4) |
| `eval_v9/judge_ablation.py` | Pairwise judging of ablation conditions (6 comparisons) |
| `eval_v9/entropy_profile.py` | Token-by-token Shannon entropy traces, JSD between routed and raw |
| `eval_v9/plot_entropy.py` | Violin plots, trajectory curves, JSD box plots |
| `eval_v9/results/` | ablation_responses.jsonl, entropy_profiles.jsonl, entropy_summary.json |

### Question Sources
| Model | Questions | Format |
|-------|-----------|--------|
| Claude Opus | 105 | Base set — mission brief format with IDs |
| DeepSeek | 102 | Extended with categories A-L, difficulty, notes |
| Gemini | 20 | Dense, high-quality boundary questions |
| GPT | 100 | Full coverage with rich notes |
| Grok | 20 | Concise, targeted |
