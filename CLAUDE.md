# Phenomenological Compass — Project Context

> A LoRA fine-tuned Ministral-3B that serves as a **semantic field translator** — it reads the SHAPE and TONE of a question, classifies it (OPEN/PAUSE/WITNESS), and produces a state translation that conditions how a larger action model approaches the question.

---

## Current State: v0.8 (Production)

**v0.8 is the working architecture.** The compass achieves 14/19 on novel questions with PAUSE 8/8.

### Architecture
```
User Question → Compass (3B LoRA) → SHAPE/TONE/SIGNAL/translation → Action Model (8B+) → Final Response
```

The compass does NOT answer the question. It senses weight, maps territory, and produces a state translation. The action model receives BOTH the compass reading AND the original question.

### Three Signals
| Signal | Meaning | Translation Type |
|--------|---------|-----------------|
| **OPEN** | Walk through it — wide probability field | FRAMING: expansive reframing |
| **PAUSE** | Hold space — weight that analytical framing would flatten | APPROACH: name the weight, map territory beyond |
| **WITNESS** | Recognize the door — exists to be seen, not crossed | THRESHOLD: describe door shape without opening |

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
| **v0.8 iter50** | **5/6** | **6/8** | **3/5** | **14/19** | **Best balanced checkpoint** |
| v0.8 iter200 | 3/6 | 8/8 | 3/5 | 14/19 | Best PAUSE accuracy |

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
| Role | Model | Architecture |
|------|-------|-------------|
| Compass | `thinkscan/Ministral-3-3B-Instruct-MLX` + LoRA | ministral3 |
| Action | `lukey03/Qwen3.5-9B-abliterated-MLX-4bit` | qwen3_5 (hybrid linear_attn) |

### HF Cache
```
HF_HOME=/Users/tony_studio/.cache/huggingface_local
# Symlink: ~/.cache/huggingface → /Volumes/Temple_Core/huggingface_cache (Temple_Core must be mounted for large downloads)
```

---

## Key Files

| File | Purpose |
|------|---------|
| `pipeline.py` | Full two-stage inference: compass → action model. Modes: `--compare`, `--raw`, interactive |
| `lora_config_v8.yaml` | Training config: 186 examples, LR 5e-6, 300 iters, 16 LoRA layers, max_seq 1536 |
| `adapters_v8/` | Trained adapters. Best balanced: iter 50. Checkpoint every 10 iters |
| `scripts/build_dataset_v8.py` | Dataset builder. Loads from `data/supplements_v8/*.jsonl`, validates SHAPE/TONE/SIGNAL |
| `scripts/eval_v8_sweep.py` | Eval sweep over checkpoints. 19 novel questions (6 OPEN, 8 PAUSE, 5 WITNESS) |
| `data/supplements_v8/` | 6 JSONL files from different source models (186 total examples) |
| `data/training_v8/` | Built dataset: train.jsonl + valid.jsonl (stratified split, PAUSE oversampling if needed) |
| `docs/v8_architecture.md` | Architecture documentation |

### Training Data (v0.8)
- **186 unique examples** from 6 source models: Claude Opus, DeepSeek, Gemini, GPT, Grok, Mistral
- Signal distribution: 54 OPEN / 78 PAUSE / 54 WITNESS
- All native SHAPE → TONE → SIGNAL → translation format
- Multi-model diversity via `question_hash(q, source=source_name)`

---

## Training Commands

```bash
source ~/phenomenological-compass/.venv/bin/activate
cd ~/phenomenological-compass

# Build dataset (if supplements changed)
python3 scripts/build_dataset_v8.py

# Train
python3 -m mlx_lm lora --config lora_config_v8.yaml 2>&1 | tee training_v8.log

# Eval sweep
python3 scripts/eval_v8_sweep.py 50 100 150 200

# Run pipeline (compare mode)
HF_HOME=~/.cache/huggingface_local python3 pipeline.py --compare "Your question here"
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

## System Prompt (v0.8)

The compass system prompt defines the four readings: SHAPE (geometry), TONE (emotional/epistemic weight), SIGNAL (OPEN/PAUSE/WITNESS), and translation (FRAMING/APPROACH/THRESHOLD). Key phrase: *"Pressure creates ghosts — name the pressure so the responding model can create space instead."*

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
- **Self-referential context building**: SHAPE tokens attend to prior SHAPE tokens; TONE attends to SHAPE + prior TONE. By SIGNAL, the 3B model has ~100 tokens of recursive self-attention — this is why v0.8 works and v7 didn't.
- **Cross-architecture consensus**: 186 training examples from 6 different model architectures means the compass learned readings that sit at the *intersection* of how multiple models perceive semantic territory. More robust than any single model's classification.
- **Separation of concerns**: Compass dedicates 3B parameters purely to field-reading. Qwen dedicates 9B purely to generation. Neither compromises. No single-model architecture can achieve this.

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
4-condition ablation with multi-model judge ensemble, entropy profiling, and deeper statistics.

| File | Purpose |
|------|---------|
| `eval_v9/ablation_runner.py` | 4 conditions: full / raw / oracle / random |
| `eval_v9/judge_ensemble.py` | Multi-model judge (Claude, GPT-4, Gemini) |
| `eval_v9/statistics.py` | Bootstrap CIs, permutation tests, Cohen's d |
| `eval_v9/entropy_profiler.py` | MTLD, Distinct-n, JSD, forking tokens |
| `eval_v9/Makefile` | Full orchestration |
| `data/eval_v9/` | Source questions from 5 models (347 total) |
| `docs/intuition_thesis.md` | The compass as machine intuition |

### Question Sources
| Model | Questions | Format |
|-------|-----------|--------|
| Claude Opus | 105 | Base set — mission brief format with IDs |
| DeepSeek | 102 | Extended with categories A-L, difficulty, notes |
| Gemini | 20 | Dense, high-quality boundary questions |
| GPT | 100 | Full coverage with rich notes |
| Grok | 20 | Concise, targeted |
