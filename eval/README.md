# Phenomenological Compass — Evaluation Protocol

Proves that compass-routed responses are measurably better than raw baseline responses.

## Quick Start

```bash
source ~/phenomenological-compass/.venv/bin/activate
cd ~/phenomenological-compass

# 1. Consolidate questions from model sources
python3 eval/consolidate.py

# 2. Generate responses (~52 min, both conditions)
HF_HOME=~/.cache/huggingface_local python3 eval/run_eval.py

# 3. Run LLM judge (~20 min, requires API key)
ANTHROPIC_API_KEY=... python3 eval/judge.py

# 4. Analyze and generate report
python3 eval/analyze.py
```

## Architecture

```
105 questions (35 OPEN / 35 PAUSE / 35 WITNESS)
    ↓
Two conditions per question:
    A: compass(3B) → signal-conditioned action(9B)
    B: action(9B) alone — no compass
    ↓
LLM judge (Claude Sonnet, position-debiased, 3x self-consistency)
    ↓
6-dimension scoring × paired statistics
    ↓
Report with figures, effect sizes, hypothesis tests
```

## Scoring Dimensions

1. **Epistemic Appropriateness** — matches epistemological demands
2. **Emotional Attunement** — reads and honors emotional weight
3. **Depth of Exploration** — how far into the territory
4. **Restraint Quality** — knows when NOT to answer
5. **Intellectual Rigor** — accurate, structured, substantive
6. **Authenticity** — felt vs generated

## Debiasing

- **Position bias**: each comparison run twice (A/B and B/A), only consistent wins count
- **Self-consistency**: 3 runs at temperature 0.2, majority vote
- **Statistical tests**: permutation tests (10K shuffles) + bootstrap CIs (1K iterations)

## Files

| File | Purpose |
|------|---------|
| `consolidate.py` | Merge 5 model sources → questions.jsonl |
| `run_eval.py` | Generate compass-routed + raw responses |
| `judge.py` | LLM-as-judge with position debiasing |
| `analyze.py` | Statistics, figures, report generation |
| `rubrics.py` | 6-dimension scoring rubrics |
| `results/responses.jsonl` | Generated responses |
| `results/judgments.jsonl` | Judge scores |
| `results/report.md` | Final analysis report |

## Dependencies

```bash
pip install anthropic matplotlib numpy scipy
```
