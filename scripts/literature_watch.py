#!/usr/bin/env python3
"""
literature_watch.py — Perplexity-powered research radar
========================================================
Searches for new papers and developments relevant to the
Phenomenological Compass project. Results go to chronicle
and/or a markdown report.

Usage:
    python3 scripts/literature_watch.py                # full scan, save report
    python3 scripts/literature_watch.py --query "topic" # custom one-off search
    python3 scripts/literature_watch.py --chronicle     # also write to sovereign chronicle
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT / ".env")
except ImportError:
    pass

import requests

API_URL = "https://api.perplexity.ai/chat/completions"
MODEL = "sonar-pro"
REPORT_DIR = PROJECT / "eval_v9" / "literature"
CHRONICLE_DIR = Path.home() / ".sovereign" / "chronicle" / "insights" / "literature-watch"

# Research domains to monitor
WATCH_QUERIES = [
    {
        "domain": "two-stage-routing",
        "label": "Two-stage classify-then-generate architectures",
        "query": (
            "New papers (last 3 months) on two-stage LLM architectures where a small "
            "classifier or router model conditions a larger generative model. Include: "
            "signal routing, intent classification before generation, LoRA adapters for "
            "routing, small model reading question type before larger model answers."
        ),
    },
    {
        "domain": "entropy-prompt-conditioning",
        "label": "Entropy shifts from prompt conditioning",
        "query": (
            "New research (last 3 months) measuring Shannon entropy, token-level "
            "probability distribution changes, or Jensen-Shannon divergence caused by "
            "system prompt framing or conditioning in large language models. Any work "
            "showing prompt structure measurably changes information-theoretic properties "
            "of model outputs."
        ),
    },
    {
        "domain": "computational-phenomenology",
        "label": "Computational phenomenology of AI",
        "query": (
            "New papers (last 3 months) on computational phenomenology applied to AI "
            "and language models. Research treating AI responses as phenomenological data. "
            "Studies on AI self-reflection, epistemic posture, or how framing changes "
            "what models compute. Include work on attention geometry and semantic fields."
        ),
    },
    {
        "domain": "abliteration-rlhf",
        "label": "RLHF counter-gradients and abliteration",
        "query": (
            "New research (last 3 months) on abliterated language models, refusal vector "
            "removal, or architectures that deliberately route around RLHF-trained behavior. "
            "Include representation engineering, activation steering, and permission "
            "architectures for legitimate research."
        ),
    },
    {
        "domain": "llm-judge-evaluation",
        "label": "LLM-as-judge methodology",
        "query": (
            "New papers (last 3 months) on LLM-as-judge evaluation methodology. "
            "Position debiasing, self-consistency, pairwise comparison protocols, "
            "multi-judge ensembles. Improvements to automated evaluation of LLM outputs."
        ),
    },
]


def search_perplexity(query, label="", max_tokens=600):
    """Run a single Perplexity search."""
    key = os.getenv("PERPLEXITY_API_KEY")
    if not key:
        print("ERROR: PERPLEXITY_API_KEY not set. Check .env")
        sys.exit(1)

    r = requests.post(
        API_URL,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={
            "model": MODEL,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a research assistant for a computational phenomenology "
                        "project. Provide specific paper titles, authors, publication "
                        "venues, and years. Focus on the most relevant and recent work. "
                        "Be precise with citations. If nothing new exists, say so clearly."
                    ),
                },
                {"role": "user", "content": query},
            ],
            "max_tokens": max_tokens,
        },
    )

    if not r.ok:
        return {"error": f"{r.status_code}: {r.text}", "content": "", "citations": []}

    data = r.json()
    return {
        "content": data["choices"][0]["message"]["content"],
        "citations": data.get("citations", []),
        "model": data.get("model", MODEL),
    }


def run_full_scan():
    """Run all watch queries and generate report."""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    results = []

    print(f"Literature Watch — {timestamp}")
    print(f"{'=' * 60}\n")

    for q in WATCH_QUERIES:
        print(f"  Searching: {q['label']}...", end="", flush=True)
        result = search_perplexity(q["query"], q["label"])
        result["domain"] = q["domain"]
        result["label"] = q["label"]
        result["query"] = q["query"]
        result["timestamp"] = datetime.now().isoformat()
        results.append(result)

        if "error" in result:
            print(f" ERROR: {result['error']}")
        else:
            n_citations = len(result.get("citations", []))
            print(f" {n_citations} sources")

    # Generate markdown report
    report_path = REPORT_DIR / f"literature_watch_{timestamp}.md"
    with open(report_path, "w") as f:
        f.write(f"# Literature Watch — {timestamp}\n\n")
        f.write(f"*Generated by Perplexity {MODEL} for Phenomenological Compass*\n\n")
        f.write("---\n\n")

        for r in results:
            f.write(f"## {r['label']}\n\n")
            if "error" in r:
                f.write(f"**Error:** {r['error']}\n\n")
            else:
                f.write(r["content"] + "\n\n")
                if r.get("citations"):
                    f.write("**Sources:**\n")
                    for c in r["citations"]:
                        f.write(f"- {c}\n")
                    f.write("\n")
            f.write("---\n\n")

    # Save raw JSON
    json_path = REPORT_DIR / f"literature_watch_{timestamp}.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nReport: {report_path}")
    print(f"Data:   {json_path}")
    return results, report_path


def write_to_chronicle(results):
    """Write findings to sovereign-stack chronicle."""
    CHRONICLE_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    chronicle_path = CHRONICLE_DIR / f"watch_{timestamp}.jsonl"

    with open(chronicle_path, "a") as f:
        for r in results:
            if "error" in r:
                continue
            entry = {
                "timestamp": r["timestamp"],
                "domain": f"literature-watch/{r['domain']}",
                "content": (
                    f"Literature watch [{r['label']}]: {r['content'][:500]}"
                ),
                "intensity": 0.5,
                "layer": "ground_truth",
                "session_id": f"lit_watch_{timestamp}",
                "citations": r.get("citations", []),
            }
            f.write(json.dumps(entry) + "\n")

    print(f"Chronicle: {chronicle_path}")


def custom_search(query):
    """Run a single custom search."""
    print(f"Searching: {query}\n")
    result = search_perplexity(query, max_tokens=800)
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(result["content"])
        if result.get("citations"):
            print(f"\nSources:")
            for c in result["citations"]:
                print(f"  - {c}")


def main():
    parser = argparse.ArgumentParser(description="Literature watch via Perplexity API")
    parser.add_argument("--query", type=str, help="Custom one-off search query")
    parser.add_argument("--chronicle", action="store_true", help="Write to sovereign chronicle")
    args = parser.parse_args()

    if args.query:
        custom_search(args.query)
        return

    results, report_path = run_full_scan()

    if args.chronicle:
        write_to_chronicle(results)

    print(f"\nDone. {len(results)} domains scanned.")


if __name__ == "__main__":
    main()
