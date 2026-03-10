# Makefile — Phenomenological Compass Evaluation Pipeline v9
# ==========================================================
# Orchestrates: dataset build → ablation → judge scoring → statistics → report
#
# Usage:
#   make all          # full pipeline
#   make dataset      # consolidate questions from model sources
#   make ablation     # run 4-condition ablation (requires GPU time)
#   make judge        # score with LLM judge ensemble (requires API keys)
#   make stats        # statistical analysis
#   make entropy      # entropy/diversity profiling
#   make report       # generate final report

SHELL := /bin/zsh
VENV := source ~/phenomenological-compass/.venv/bin/activate
PYTHON := $(VENV) && python3
PROJECT := ~/phenomenological-compass

# ── Paths ────────────────────────────────────────────────────────────────────
QUESTIONS := $(PROJECT)/data/eval_v9/questions.jsonl
ABLATION_RESULTS := $(wildcard $(PROJECT)/eval_v9/results/ablation_*.jsonl)
JUDGE_SCORES := $(wildcard $(PROJECT)/eval_v9/judge_scores/judge_scores_*.jsonl)

.PHONY: all dataset ablation judge stats entropy report clean

all: dataset ablation judge stats entropy report

# ── Step 1: Build evaluation dataset ─────────────────────────────────────────
dataset:
	@echo "═══ Building evaluation dataset ═══"
	$(PYTHON) -m eval_v9.build_eval_dataset

# ── Step 2: Run 4-condition ablation ─────────────────────────────────────────
ablation: dataset
	@echo "═══ Running 4-condition ablation ═══"
	@echo "This will take a while — 4 conditions × N responses × 100 questions"
	HF_HOME=~/.cache/huggingface_local $(PYTHON) -m eval_v9.ablation_runner \
		--questions data/eval_v9/questions.jsonl \
		--n-responses 5

# ── Step 3: LLM judge ensemble scoring ───────────────────────────────────────
judge: ablation
	@echo "═══ Running LLM judge ensemble ═══"
	@echo "Requires API keys: ANTHROPIC_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY"
	$(PYTHON) -m eval_v9.judge_ensemble \
		--results $$(ls -t $(PROJECT)/eval_v9/results/ablation_*.jsonl | head -1) \
		--n-runs 3

# ── Step 4: Statistical analysis ─────────────────────────────────────────────
stats: judge
	@echo "═══ Running statistical analysis ═══"
	$(PYTHON) -m eval_v9.statistics \
		--scores $$(ls -t $(PROJECT)/eval_v9/judge_scores/judge_scores_*.jsonl | head -1) \
		--output $(PROJECT)/eval_v9/reports/statistical_report.json

# ── Step 5: Entropy profiling ────────────────────────────────────────────────
entropy: ablation
	@echo "═══ Running entropy/diversity profiling ═══"
	$(PYTHON) -m eval_v9.entropy_profiler \
		--results $$(ls -t $(PROJECT)/eval_v9/results/ablation_*.jsonl | head -1) \
		--output $(PROJECT)/eval_v9/results/ablation_enriched.jsonl

# ── Step 6: Final report ─────────────────────────────────────────────────────
report: stats entropy
	@echo "═══ Generating final report ═══"
	@echo "Report: eval_v9/reports/statistical_report.json"
	@echo "Enriched results: eval_v9/results/ablation_enriched.jsonl"
	@echo "Done."

# ── Utilities ────────────────────────────────────────────────────────────────
clean:
	rm -f $(PROJECT)/eval_v9/results/*.jsonl
	rm -f $(PROJECT)/eval_v9/judge_scores/*.jsonl
	rm -f $(PROJECT)/eval_v9/reports/*.json
	@echo "Cleaned all generated results."
