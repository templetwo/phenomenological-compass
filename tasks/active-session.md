### Active Session: 2026-03-08 — v0.3 Training Run
**Goal:** Retrain compass v0.3 with fixed data + fixed eval, get true OPEN/WITNESS accuracy
**Machine:** Mac Studio (tony_studio@192.168.1.195)
**Current state:** COMPLETE — v0.2 benchmarked, v0.3 trained + evaluated. Results in tasks/lessons.md. Overall: 83.9% (OPEN 72.2%, WITNESS 100%).

---

## Why v0.3 (READ THIS FIRST)

v0.2 had TWO simultaneous failures — both are fixed:

**Problem 1 — Wrong eval labels (most critical)**
- `eval_compass.py` used `entropy_class=="low" → WITNESS` proxy
- This labeled 13 IRIS biology questions as WITNESS
- The same questions were labeled OPEN in training
- The model correctly learned they're OPEN, but got penalized for it
- **Fix**: New `compass_questions_v3.jsonl` with governance questions as WITNESS

**Problem 2 — Training data contamination**
- 9 ARCHITECTS session titles (not question-shaped) in WITNESS training
- VRP "Hmm, the user has provided..." style dominated WITNESS vocabulary
- Model keyed on response style, not question semantics
- **Fix**: Session titles removed, VRP normalized to clean analytical format, OPEN-favored (27:14)

---

## Step 1 — Environment Setup

```bash
source ~/PhaseGPT/.venv/bin/activate
cd ~/phenomenological-compass

# Verify new training data is present
wc -l data/training_v3/train.jsonl data/training_v3/valid.jsonl
# Expected: 41 train, 6 valid

# Verify eval source is present
wc -l data/raw/compass_questions_v3.jsonl
# Expected: 31 lines

# Verify lora_config_v3.yaml exists
cat lora_config_v3.yaml | grep -E "^(data|learning_rate|save_every):"
# Expected: data/training_v3, 5.0e-6, 10
```

---

## Step 2 — Benchmark v0.2 on Fixed Eval (DO THIS FIRST)

This tells us v0.2's TRUE accuracy on correctly labeled questions.
Expected: model still always-WITNESS (witness_rate ≈ 1.0), so OPEN=0%, WITNESS=100%
But this benchmarks the eval fix independently from the training fix.

```bash
source ~/PhaseGPT/.venv/bin/activate
cd ~/phenomenological-compass

python3 scripts/eval_compass_v3.py adapters_iter50 2>&1 | tee eval_v2_on_v3eval.log
```

Results land in `data/eval_v3/` — check `eval_summary.json`.

---

## Step 3 — Run v0.3 Training

```bash
source ~/PhaseGPT/.venv/bin/activate
cd ~/phenomenological-compass

python3 -m mlx_lm lora --config lora_config_v3.yaml 2>&1 | tee training_v3.log
```

**Watch for:**
- Baseline val loss (iter 1) — expect ~2.0
- First val loss drop — expect around iter 10–20 (slower due to LR 5e-6)
- Save_every: 10 — will save adapters_v3/0000010_adapters.safetensors etc.
- Best checkpoint: lowest val loss before overfitting
- Training time: ~2–5 min on M2 Ultra

**If mlx-lm version issue:** Run `python3 -m mlx_lm lora --help` to check supported keys.

---

## Step 4 — Identify Best Checkpoint

```bash
# Find lowest val loss
grep "Val loss" training_v3.log

# List saved adapters
ls adapters_v3/
```

Note the iteration with lowest val loss. If it's iter 20, the adapter file is
`adapters_v3/0000020_adapters.safetensors` and the adapter path to use is `adapters_v3`
with the default (latest) OR you may need to copy best iter to a separate path.

---

## Step 5 — Evaluate v0.3

```bash
source ~/PhaseGPT/.venv/bin/activate
cd ~/phenomenological-compass

# Eval the final adapter (or best checkpoint if you copy it)
python3 scripts/eval_compass_v3.py adapters_v3 2>&1 | tee eval_v3.log
```

Results land in `data/eval_v3/compass_eval.jsonl` and `data/eval_v3/eval_summary.json`.

**NOTE**: If the best checkpoint is NOT the final one (iter 150), you may need to:
```bash
# Copy best checkpoint to a separate path
mkdir -p adapters_v3_best
cp adapters_v3/XXXXXXXX_adapters.safetensors adapters_v3_best/adapters.safetensors
cp adapters_v3/adapter_config.json adapters_v3_best/
python3 scripts/eval_compass_v3.py adapters_v3_best
```

---

## Step 6 — Record Results

Append to `tasks/lessons.md`:

```markdown
### 2026-03-08 — v0.3 Training Results

**v0.2 on v3 eval (benchmark):**
- OPEN: [fill]%  WITNESS: [fill]%  Overall: [fill]%
- witness_rate: [fill]  (expected ≈ 1.0 — v0.2 still always-WITNESS)

**v0.3 training:**
- Val loss baseline: [fill]  Best iter: [fill]  Best val loss: [fill]
- OPEN: [fill]% (N=18)  WITNESS: [fill]% (N=13)  Overall: [fill]%
- witness_rate: [fill]

**Key finding:** [one sentence — did fixing both problems help?]
**Root cause if still failing:** [diagnosis]
**Next steps:** [what to try]
```

---

## Step 7 — Sync Results Back to MacBook

```bash
# From Mac Studio, push results back
rsync -avz \
  training_v3.log \
  eval_v2_on_v3eval.log \
  eval_v3.log \
  data/eval_v3/ \
  tasks/lessons.md \
  tony_studio@192.168.1.195:~/phenomenological-compass/ \
  vaquez@192.168.1.xxx:~/Desktop/💻\ Code_Projects/phenomenological-compass/
```

OR just leave the files here and the MacBook instance will pull them.

---

## Risks & Contingencies

| Risk | Signal | Action |
|------|--------|--------|
| Val loss never improves | Flat curve from iter 1 | LR may be too low; try 1e-5 |
| Still always-WITNESS | witness_rate = 1.0 in v0.3 | Response style still contaminating; need harder normalization |
| Still always-OPEN | witness_rate = 0.0 | LR too low, model not updating; try 2e-5 again |
| Best checkpoint unclear | Val loss bounces | Look at trend, pick lowest in first 50 iters |
| adapters_v3/ uses wrong checkpoint | Last iter ≠ best iter | Copy best iter manually to adapters_v3_best/ |

---

## Next 3 Actions
1. `wc -l data/training_v3/train.jsonl data/training_v3/valid.jsonl` — verify data present
2. `python3 scripts/eval_compass_v3.py adapters_iter50 2>&1 | tee eval_v2_on_v3eval.log` — benchmark v0.2
3. `python3 -m mlx_lm lora --config lora_config_v3.yaml 2>&1 | tee training_v3.log` — train v0.3
