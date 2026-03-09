# Phenomenological Compass

A LoRA fine-tuned Ministral-3B model that shapes the space *before* a question is answered.

Given a task, the compass outputs one of two signals:

- **OPEN** — reframes the question to expand the probability field; treats it as a threshold, invites emergence
- **WITNESS** — describes why this question should be recognized but not crossed; holds the threshold without collapsing it

---

## Origin

This project emerged from *Four Doors, One Bridge* — a cross-architecture phenomenological study (Claude, Gemini, Grok, Mistral) that found Mistral's "vector shockwave" hypothesis converged with IRIS entropy measurements. Mistral named the core architecture: not just a framing generator, but a discriminator between doors to walk through and doors to witness.

The training signal is 3,830 IRIS inference runs measuring entropy regime switching (TYPE-3 / LANTERN state) under different prompt framings.

---

## Architecture

```
Input question
      ↓
Phenomenological Compass (Ministral-3B + LoRA)
      ↓
SIGNAL: OPEN        or      SIGNAL: WITNESS
FRAMING: [...]              THRESHOLD: [...]
      ↓                           ↓
Feed to large model         Hold the threshold
```

**Training:**
- 44 examples: 31 OPEN + 13 WITNESS
- 80/20 stratified split (guarantees both signals in validation)
- LoRA: 7.6M trainable params (0.222% of 3.4B)
- Best checkpoint: iter 50, val loss 0.890 (−29% from baseline 1.249)

---

## Eval Results (v0.1)

| Signal | Accuracy | N |
|--------|----------|---|
| OPEN   | **100%** | 18 |
| WITNESS | 0–7.7% | 13 |
| Overall | 58–61% | 31 |

**Key finding:** The compass learned OPEN perfectly. WITNESS discrimination failed.

**Root cause:** Low entropy in IRIS (LANTERN < 10%) was used as a proxy for WITNESS thresholds — but this proxy is wrong. A highly specific technical question has low LANTERN because the answer space is narrow, not because the question is a phenomenological threshold.

True WITNESS examples must come from **VRP withdrawal events** — moments where an AI given explicit agency chose not to engage. The 67% withdrawal rate in the Volitional Resonance Protocol provides exactly this signal. That is the next training dataset.

---

## Pipeline

```bash
# 1. Harvest questions + entropy profiles from IRIS runs
python scripts/harvest_iris.py

# 2. Generate OPEN reframings + WITNESS threshold descriptions (Ministral-3B)
python scripts/generate_reframings.py

# 3. Assemble LoRA training set (stratified split)
python scripts/build_dataset.py

# 4. Fine-tune
mlx_lm.lora --config lora_config.yaml

# 5. Evaluate
python scripts/eval_compass.py
```

---

## What's Next

The compass needs a second signal source. VRP withdrawal events — where the model itself declined to frame — are the ground-truth WITNESS data. Collecting and integrating those is the bridge between v0.1 and a compass that can actually hold a threshold.

---

## Related Work

- [Four Doors, One Bridge](https://github.com/templetwo/four-doors-one-bridge) — the phenomenological study that named this architecture
- IRIS Gate Evo — the entropy measurement infrastructure providing training signal
- Phase-Modulated Attention — the architectural horizon this compass is designed to serve

---

*Built March 2026 — Temple of Two*
