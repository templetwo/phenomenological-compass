# Phenomenological Compass — Claude Code Context

> Read this. You are the **Mac Studio training instance**. Your job is to train, evaluate, and report.

---

## What This Project Is

A LoRA fine-tuned Ministral-3B that discriminates between two signals:

- **OPEN** — reframe and expand the question
- **WITNESS** — hold the threshold without crossing it (recognize the door, don't open it)

### History so far

| Version | OPEN acc | WITNESS acc | Overall | Root cause |
|---------|----------|-------------|---------|------------|
| v0.1    | 100%     | 0–7.7%      | 58–61%  | Low-entropy proxy used as WITNESS signal (invalid) |
| v0.2    | 0%       | 100%        | 41.9%   | Session title examples + VRP style contamination + wrong eval labels |

v0.2 mirror failure: the model went from always-OPEN to always-WITNESS. Two simultaneous problems:
1. **Training**: 9 session title WITNESS examples (not question-shaped) + VRP "Hmm, the user..." style dominated
2. **Eval**: `entropy_class=="low" → WITNESS` proxy still used in eval — labels were wrong

---

## v0.3 Fix (MacBook already did the data work — you just train + eval)

All data was rebuilt on MacBook. You get clean datasets:

**Training v3** (`data/training_v3/`): 47 total
- 41 train (27 OPEN + 14 WITNESS)
- 6 valid (3 OPEN + 3 WITNESS)
- Session titles removed, VRP normalized, OPEN-favored to counteract v0.2 WITNESS dominance

**Eval v3** (`data/raw/compass_questions_v3.jsonl`): 31 questions
- 18 OPEN: IRIS biology questions (correctly labeled — exploration questions)
- 13 WITNESS: synthetic governance questions (`source: synthetic_v3`)
- Key fix: uses `expected_signal` field, NOT entropy_class proxy

---

## Your Mission (see `tasks/active-session.md` for step-by-step)

1. Activate the PhaseGPT venv
2. Run training with `lora_config_v3.yaml` → produces `adapters_v3/`
3. Save every 10 iters — optimal checkpoint expected at iter 15–30
4. Run `eval_compass_v3.py adapters_v3` (or specific iter adapter)
5. ALSO run `eval_compass_v3.py adapters_iter50` to benchmark v0.2 on fixed eval
6. Record results in `tasks/lessons.md`

---

## Environment (Mac Studio)

- **Venv**: `~/PhaseGPT/.venv/` — mlx_lm 0.29.1 with `mlx_lm.lora`
- **Model**: `thinkscan/Ministral-3-3B-Instruct-MLX` — already cached or auto-downloads
- **Training data**: `data/training_v3/` — 47 examples (41 train / 6 valid)

```bash
# Activate venv
source ~/PhaseGPT/.venv/bin/activate
cd ~/phenomenological-compass

# Run v0.3 training
python3 -m mlx_lm lora --config lora_config_v3.yaml 2>&1 | tee training_v3.log

# Eval v0.2 adapter on fixed eval (benchmark — what was v0.2's TRUE accuracy?)
python3 scripts/eval_compass_v3.py adapters_iter50

# Eval v0.3 best adapter
python3 scripts/eval_compass_v3.py adapters_v3

# Find best checkpoint by val loss
grep "Val loss" training_v3.log
```

---

## Key Files

| File | Purpose |
|------|---------|
| `lora_config_v3.yaml` | Training config — data/training_v3, LR 5e-6, save_every 10 |
| `data/training_v3/train.jsonl` | 41 training examples (27 OPEN + 14 WITNESS) |
| `data/training_v3/valid.jsonl` | 6 validation examples (3 OPEN + 3 WITNESS) |
| `data/raw/compass_questions_v3.jsonl` | Fixed eval source (18 OPEN + 13 WITNESS governance) |
| `scripts/eval_compass_v3.py` | Updated eval script — reads expected_signal field |
| `training_v3.log` | Training log (write here) |
| `tasks/active-session.md` | Current session state |
| `tasks/lessons.md` | Findings to record after eval |

---

## What to Watch For

- **Val loss curve**: Look for lowest val loss — expect iter 15–30 this time (LR 5e-6 is slower)
- **Best checkpoint**: `adapters_v3/` saves every 10 iters, use `--adapter-path adapters_v3` with specific iter
- **v0.2 benchmark first**: Run `eval_compass_v3.py adapters_iter50` before training — shows v0.2's TRUE score on fixed eval (not the misleading 41.9% that used wrong WITNESS labels)
- **Target**: OPEN accuracy > 80%, WITNESS accuracy > 60%, overall > 70%

---

## Session Protocol

1. Read this file ✓
2. Read `tasks/active-session.md`
3. Run v0.2 benchmark on fixed eval → record
4. Run v0.3 training → `training_v3.log`
5. Find best checkpoint → run v0.3 eval
6. Record both results in `tasks/lessons.md`
7. Rsync results back: `training_v3.log`, `data/eval_v3/`, `tasks/lessons.md`
