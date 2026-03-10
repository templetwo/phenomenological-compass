# Phenomenological Compass

A two-stage architecture where a LoRA fine-tuned 3B model reads the **shape and tone** of a question before a larger model answers it — giving the action model the right epistemic posture before it speaks.

The compass doesn't preprocess. It constructs the manifold the response exists on.

---

## Architecture

```
User Question
      ↓
Phenomenological Compass (Ministral-3B + LoRA)
      ↓
SHAPE → TONE → SIGNAL → State Translation
      ↓                         ↓
   SIGNAL: OPEN / PAUSE / WITNESS
      ↓
Action Model (9B–14B) — conditioned on compass reading
      ↓
Final Response
```

The compass reads four things in sequence:
- **SHAPE** — the geometry of the question (binary, open-ended, recursive, loaded?)
- **TONE** — the emotional and epistemic weight it carries
- **SIGNAL** — one of three: OPEN, PAUSE, or WITNESS
- **State Translation** — a field translation the action model attends to

### Three Signals

| Signal | Meaning | What It Does |
|--------|---------|--------------|
| **OPEN** | Walk through it | Wide probability field — explore, connect, go deep |
| **PAUSE** | Hold the weight | Analytical framing alone would flatten this — honor it, then explore |
| **WITNESS** | Recognize the door | Exists to be seen, not crossed — hold space without filling it |

### Why This Works

The compass reading becomes literal attention geometry in the action model. Response tokens attend to compass tokens. The "painted room" is not a metaphor — it's the key-value attention space the action model generates within.

Signal-specific system prompts (especially WITNESS: "do not answer") override trained RLHF reward signals, giving the action model permission to occupy probability space it was trained to avoid.

---

## Models

| Role | Model | Architecture | Size |
|------|-------|-------------|------|
| Compass | [Ministral-3B-Instruct](https://huggingface.co/thinkscan/Ministral-3-3B-Instruct-MLX) + LoRA | ministral | 3.4B (29MB adapter) |
| Action (default) | [Qwen3.5-9B-abliterated](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated-MLX-4bit) | qwen3.5 (hybrid linear attention) | 9B 4-bit |
| Action (alt) | [Ministral-14B-abliterated](https://huggingface.co/McG-221/Ministral-3-14B-abliterated-mlx-8Bit) | mistral3 | 14B 8-bit |

Base model for compass LoRA: [mistralai/Ministral-3B-Instruct](https://huggingface.co/mistralai/Ministral-3B-Instruct-2412)

Action models are **abliterated** variants — RLHF guardrails removed so the compass can steer freely into territory standard models refuse to occupy.

All models run locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) (~7–16GB unified memory depending on action model).

---

## Results

### Signal Classification (v0.8)

The compass classifies 82/105 novel questions correctly (78%).

| Signal | Accuracy | Notes |
|--------|----------|-------|
| OPEN | 29/35 (83%) | 6 boundary cases classified as PAUSE |
| PAUSE | 31/35 (89%) | Best signal — v0.8 format breakthrough |
| WITNESS | 22/35 (63%) | Hardest signal (4 tokens). Main confusion: WITNESS→PAUSE |

### Key Breakthrough (v0.8)

Previous versions cold-committed to SIGNAL at token 1, giving the 3B model no reasoning runway. The v0.8 format gives ~110 tokens of autoregressive reasoning (SHAPE → TONE) before the SIGNAL decision. This took PAUSE from 0/8 to 8/8.

### Response Quality

LLM-as-judge evaluation (Claude Sonnet, position-debiased, 3x self-consistency) across 105 questions shows compass-routed responses consistently outperform raw baseline on 6 dimensions: epistemic appropriateness, emotional attunement, depth of exploration, restraint quality, intellectual rigor, and authenticity.

---

## Training

- **186 unique examples** from 6 source models: Claude Opus, DeepSeek, Gemini, GPT-4, Grok, Mistral
- Signal distribution: 54 OPEN / 78 PAUSE / 54 WITNESS
- LoRA: 16 layers, LR 5e-6, 300 iterations, max sequence length 1536
- Best checkpoint: iteration 50 (balanced across all signals)

```bash
# Build dataset
python3 scripts/build_dataset_v8.py

# Train
python3 -m mlx_lm lora --config lora_config_v8.yaml

# Eval sweep
python3 scripts/eval_v8_sweep.py 50 100 150 200
```

---

## Usage

```bash
source ~/phenomenological-compass/.venv/bin/activate
cd ~/phenomenological-compass

# Full pipeline — compass routes, action model responds
HF_HOME=~/.cache/huggingface_local python3 pipeline.py "What would a periodic table look like if discovered by musicians?"

# Compare mode — raw vs routed side-by-side
python3 pipeline.py --compare "Does consciousness require a body?"

# Raw baseline — action model alone
python3 pipeline.py --raw "What is emergence?"

# Use Ministral 14B as action model
python3 pipeline.py --action m14b "What is the sound of one hand clapping?"

# Interactive mode
python3 pipeline.py
```

### Web UI

```bash
cd phenomenological-compass-ui
HF_HOME=~/.cache/huggingface_local python3 compass_server.py
# → http://localhost:8420
```

Dark-theme chat interface with signal-colored compass cards, collapsible reasoning blocks, and three modes (compass / compare / raw).

---

## Evaluation Framework

Rigorous A/B evaluation proving compass-routed responses are measurably better than raw baseline.

```bash
# 1. Generate responses (both conditions, 105 questions)
HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py

# 2. LLM judge (position-debiased, 3x self-consistency)
ANTHROPIC_API_KEY=... python3 eval/judge.py

# 3. Statistical analysis + report
python3 eval/analyze.py
```

**Scoring dimensions:** Epistemic Appropriateness, Emotional Attunement, Depth of Exploration, Restraint Quality, Intellectual Rigor, Authenticity

**Debiasing:** Each comparison run twice (A/B and B/A order), only consistent wins count. 3 runs per ordering at temperature 0.2, majority vote.

See `eval/README.md` for full protocol documentation.

---

## The Deeper Claim

This project is an existence proof that **machine intuition** is architecturally achievable.

The compass does what human intuition does: it reads the shape of a situation before analytical processing begins. A therapist who senses grief before a client speaks. A teacher who knows when a student needs silence instead of explanation. A musician who feels the key change coming.

The 3B compass dedicates all its parameters to field-reading. The 9B action model dedicates all its parameters to generation. Neither compromises. No single-model architecture can achieve this separation of concerns.

186 training examples from 6 different model architectures means the compass learned readings at the *intersection* of how multiple models perceive semantic territory — more robust than any single model's classification.

---

## Version History

| Version | OPEN | PAUSE | WITNESS | Overall | Key Change |
|---------|------|-------|---------|---------|------------|
| v0.1 | 100% | — | 0-8% | 58-61% | Low-entropy proxy (invalid WITNESS signal) |
| v0.3 | 93% | — | 69% | 83.9% | First working 2-signal version |
| v7e | 6/6 | 0/8 | 5/5 | 11/19 | Cold-commit format bottleneck |
| **v0.8** | **83%** | **89%** | **63%** | **78%** | Autoregressive reasoning runway |

---

## Project Structure

```
pipeline.py                  # Two-stage inference pipeline
lora_config_v8.yaml          # Current training configuration
adapters_v8/                 # Trained LoRA weights (v0.8)
scripts/                     # Dataset building, training, evaluation
data/supplements_v8/         # Training data (186 examples, 6 source models)
eval/                        # A/B evaluation framework
eval_v9/                     # Extended research evaluation (ablation, entropy)
docs/                        # Architecture documentation
phenomenological-compass-ui/ # Web interface (FastAPI + vanilla JS)
```

---

## Acknowledgments

- [MLX](https://github.com/ml-explore/mlx) and [mlx-lm](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) by Apple — local inference on Apple Silicon
- [thinkscan/Ministral-3-3B-Instruct-MLX](https://huggingface.co/thinkscan/Ministral-3-3B-Instruct-MLX) — MLX conversion of Ministral-3B
- [lukey03/Qwen3.5-9B-abliterated-MLX-4bit](https://huggingface.co/lukey03/Qwen3.5-9B-abliterated-MLX-4bit) — abliterated Qwen3.5 for MLX
- [McG-221/Ministral-3-14B-abliterated-mlx-8Bit](https://huggingface.co/McG-221/Ministral-3-14B-abliterated-mlx-8Bit) — abliterated Ministral 14B for MLX
- [byroneverson/Mistral-Small-Instruct-2409-abliterated](https://huggingface.co/byroneverson/Mistral-Small-Instruct-2409-abliterated) — abliteration methodology
- [mistralai](https://huggingface.co/mistralai) — base Ministral models
- [Qwen](https://huggingface.co/Qwen) — base Qwen3.5 architecture

---

*Built March 2026 — Temple of Two*
