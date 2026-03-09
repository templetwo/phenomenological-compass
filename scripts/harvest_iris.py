"""
harvest_iris.py
===============
Extract questions + entropy profiles from all IRIS runs.
Outputs: data/raw/iris_questions.jsonl

Each record:
  {
    "run_id":       str,
    "question":     str,
    "domain":       str,
    "jsd":          float,
    "cosine":       float,
    "type_dist":    {"0": f, "1": f, "2": f, "3": f},
    "lantern_pct":  float,   # TYPE-3 share
    "entropy_class": str,    # "high" | "mid" | "low"
    "converged":    bool,
  }
"""

import json
import glob
import os

IRIS_ROOTS = [
    "/Users/vaquez/Iris-Gate-Evo/runs",
    "/Users/vaquez/iris-evo-findings/runs",
]

OUT = "/Users/vaquez/Desktop/💻 Code_Projects/phenomenological-compass/data/raw/iris_questions.jsonl"

def entropy_class(lantern_pct: float) -> str:
    if lantern_pct >= 0.25:
        return "high"
    elif lantern_pct >= 0.10:
        return "mid"
    else:
        return "low"

def extract_domain(run_id: str) -> str:
    # run_id format: evo_YYYYMMDD_HHMMSS_domain(+domain)
    parts = run_id.split("_")
    return parts[-1] if len(parts) >= 4 else "unknown"

records = []
seen = set()

for root in IRIS_ROOTS:
    pkg_paths = glob.glob(os.path.join(root, "*/full_package.json"))
    for pkg_path in pkg_paths:
        run_dir = os.path.dirname(pkg_path)
        run_id  = os.path.basename(run_dir)

        if run_id in seen:
            continue
        seen.add(run_id)

        try:
            pkg = json.load(open(pkg_path))
            s1_path = os.path.join(run_dir, "s1_formulations.json")
            if not os.path.exists(s1_path):
                continue
            s1 = json.load(open(s1_path))

            question = pkg.get("question", "").strip()
            if not question:
                continue

            snap      = s1.get("snapshot", {})
            type_dist = snap.get("type_distribution", {})
            lantern   = float(type_dist.get("3", 0.0))
            jsd       = snap.get("jsd")
            cosine    = snap.get("cosine")

            conv_report = pkg.get("convergence_report", {})
            converged   = bool(conv_report.get("passed", False)) \
                          if isinstance(conv_report, dict) else False

            records.append({
                "run_id":        run_id,
                "question":      question,
                "domain":        extract_domain(run_id),
                "jsd":           jsd,
                "cosine":        cosine,
                "type_dist":     {k: float(v) for k, v in type_dist.items()},
                "lantern_pct":   lantern,
                "entropy_class": entropy_class(lantern),
                "converged":     converged,
            })

        except Exception as e:
            print(f"  [SKIP] {run_id}: {e}")

# Sort by lantern_pct descending
records.sort(key=lambda r: r["lantern_pct"], reverse=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    for r in records:
        f.write(json.dumps(r) + "\n")

# Summary
high = sum(1 for r in records if r["entropy_class"] == "high")
mid  = sum(1 for r in records if r["entropy_class"] == "mid")
low  = sum(1 for r in records if r["entropy_class"] == "low")

print(f"Harvested {len(records)} runs → {OUT}")
print(f"  HIGH (LANTERN ≥ 25%): {high}")
print(f"  MID  (10–25%):        {mid}")
print(f"  LOW  (<10%):          {low}")
print(f"  Mean LANTERN:         {sum(r['lantern_pct'] for r in records)/len(records):.3f}")
