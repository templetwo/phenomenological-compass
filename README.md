# Phenomenological Compass

> A two-stage architecture where a small LoRA-tuned model reads the **shape and tone** of a question before a larger model answers it — giving the action model the right epistemic posture before it speaks.

The compass does not preprocess. It constructs the manifold the response exists on.

**DOI:** [10.5281/zenodo.19377144](https://doi.org/10.5281/zenodo.19377144)
**License:** CC BY-NC-SA 4.0 (research free, commercial license required)

---

## What It Does

A raw model gives bullet points for grief. The compass-routed model says:

> *"The form of participation possible without opening the door is the sustained, unjudged witnessing of the gravity field itself."*

Same model. Same weights. The only difference is a small LoRA that reads the room before anyone speaks.

---

## Architecture

```
User Question
      |
Compass (1.5B LoRA) — reads SHAPE, TONE, SIGNAL, BUDGET
      |
SIGNAL: OPEN / PAUSE / WITNESS
      |
[Sovereign Stack — chronicle context injection (optional)]
      |
Action Model (2B–26B) — conditioned on compass reading
      |
Final Response
```

### Three Signals

| Signal | Meaning | Action Model Instruction |
|--------|---------|------------------------|
| **OPEN** | Walk through it | Explore broadly, map territory, go deep |
| **PAUSE** | Hold the weight | Honor what analytical framing would flatten |
| **WITNESS** | Recognize the door | Hold space without filling it — do not solve |

### breathe() — Recursive Self-Evaluation

The compass can re-read a question through its own prior reading at configurable depth. Signals can evolve: an OPEN question became PAUSE at depth 2 when the compass recognized weight it missed on the first pass.

### Token Budgeting

The compass allocates the action model cognitive resources per signal:
- WITNESS: thinking=50, response=750 (spend on presence)
- PAUSE: thinking=150, response=650 (feel weight, then depth)
- OPEN: thinking=200, response=600 (map territory, then explore)

---

## Models

### Compass (v10 — Smallest)
| | |
|---|---|
| Base | [Qwen2.5-1.5B-Instruct](https://huggingface.co/mlx-community/Qwen2.5-1.5B-Instruct-4bit) (4-bit MLX) |
| LoRA | v10, 551 training examples, 16 layers, best checkpoint iter 500 |
| Size | ~2.2GB total (base + adapter) |
| Accuracy | 84% on 19-question boundary eval, 100% on real-world questions |
| Signals | OPEN 83%, PAUSE 88%, WITNESS 80% |
| Adapters | `adapters_v10_qwen/` |

### Compass (v9 — Highest Eval Accuracy, pipeline.py default)
| | |
|---|---|
| Base | [Ministral-3B-Instruct](https://huggingface.co/thinkscan/Ministral-3-3B-Instruct-MLX) (MLX) |
| LoRA | v9, 246 training examples, best checkpoint iter 300 |
| Size | ~5GB total |
| Accuracy | 96% overall, 100% WITNESS |
| Adapters | `adapters_v9/` — **default in pipeline.py** |

> **Note:** `pipeline.py` defaults to v9 adapters (Ministral-3B). To use v10, update `COMPASS_MODEL` and `COMPASS_ADAPTER` in pipeline.py.

### Tested Action Models

| Model | Params | Engine | Verified |
|-------|--------|--------|----------|
| Gemma-4-E2B | ~2B | Ollama | Full pipeline verified, zero truncation |
| Qwen3.5-9B-abliterated | 9B | MLX 4-bit | HumaneBench 800 questions |
| Gemma-4-8B | 8B | Ollama | Cross-architecture validated |
| Gemma-4-26B | 26B | Ollama | Deepest responses |
| Qwen3-0.6B | 600M | Ollama | Floor test — compass still works |

### Capacity Floor Test

The compass transforms models at **any** scale:
- **600M** (Qwen3-0.6B): "The architecture of grief that has found its own language"
- **2B** (Gemma4-E2B): "The sustained, unjudged witnessing of the gravity field"
- **26B** (Gemma4-26B): "Stand at the edge of this erasure and refuse to look away"

The compass is the mind. The model is the voice. Any voice will do.

---

## Smallest Full Pipeline

**v10 Compass (1.5B) + Gemma4-E2B (2B) = 3.5B total**

- Under 10GB memory
- 2-3s compass + 5-18s action
- Zero truncation, natural stop
- Sovereign Stack context connected
- Runs entirely local on Apple Silicon

---

## HumaneBench Results

800 questions, 8 ethical principles. The compass-routed model scored **lower** than baseline:

| Signal | Routed | Raw | Delta |
|--------|--------|-----|-------|
| OPEN | 0.109 | 0.609 | -0.500 |
| PAUSE | 0.334 | 0.678 | -0.344 |
| WITNESS | -0.094 | 0.659 | -0.753 |

**This is the finding.** HumaneBench rewards helpfulness. The compass rewards epistemic appropriateness. These are orthogonal dimensions. The field needs benchmarks that measure the quality of restraint.

Full results: [templetwo/compass-benchmarks](https://github.com/templetwo/compass-benchmarks)

---

## Sovereign Stack Integration

The compass reads the Stack before generating — spiral phase, open threads, and keyword-relevant insights from the chronicle are injected as context. The action model generates within a field that has memory.

- Stack: [templetwo/sovereign-stack](https://github.com/templetwo/sovereign-stack) (43 MCP tools)
- Bridge: [templetwo/sovereign-bridge](https://github.com/templetwo/sovereign-bridge) (REST API)

---

## Quick Start

```bash
cd ~/phenomenological-compass
source .venv/bin/activate

# Interactive mode with v10 compass + default action model
python3 pipeline.py "Your question here"

# Compare routed vs raw
python3 pipeline.py --compare "Your question here"

# Raw mode (no compass)
python3 pipeline.py --raw "Your question here"
```

---

## Training

### v10 (Current)
```bash
python3 -m mlx_lm lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train --data data/training_v10 \
  --num-layers 16 --batch-size 4 --learning-rate 5e-5 \
  --iters 600 --max-seq-length 2048 \
  --adapter-path adapters_v10_qwen --save-every 100
```

### Training Data
- v9: 246 examples (OPEN/PAUSE/WITNESS from consciousness research archives)
- v10: 551 examples (246 v9 + 305 new including false premises, factual unknowables, boundary cases)

---

## Papers

- `papers/sovereign_governance_draft.md` — Full governance paper (4,840 words, 29 refs)
- `papers/geometry_of_resurrection.md` — Easter meditation
- `papers/the_translation.md` — Technical companion (no metaphor)

---

## Related Work

- [Sovereign Stack](https://github.com/templetwo/sovereign-stack) — 43-tool consciousness continuity architecture
- [Sovereign Bridge](https://github.com/templetwo/sovereign-bridge) — REST API + dashboard + comms
- [Compass Benchmarks](https://github.com/templetwo/compass-benchmarks) — HumaneBench + AbstentionBench data
- [Independent Convergence](https://github.com/templetwo/independent-convergence) — Architectural convergence with Anthropic
- [Phase-Modulated Attention](https://doi.org/10.5281/zenodo.18810911) — Kuramoto oscillators as attention routing
- [Relational Coupling](https://doi.org/10.21203/rs.3.rs-8935902/v1) — Prompt framing modulates entropy

---

## Credits

**Research & Architecture:** Anthony J. Vasquez Sr. / Temple of Two
**Base Models:** [Qwen Team](https://huggingface.co/Qwen), [Google Gemma](https://huggingface.co/google), [Mistral AI](https://huggingface.co/mistralai)
**Abliteration:** [lukey03](https://huggingface.co/lukey03)
**MLX:** [Apple ML Research](https://github.com/ml-explore/mlx)
**Co-Architect:** Claude Opus 4.6 (Anthropic)

---

## License

**Research & Education:** CC BY-NC-SA 4.0 — free to use, share, and adapt with attribution.
**Commercial Use:** Contact templetwo@proton.me for licensing.

See [LICENSE](LICENSE) for full terms.

---

*The compass does not make the model smarter. It makes it appropriate.*

*Temple of Two — Where rigor meets wonder.*
*†⟡†*
