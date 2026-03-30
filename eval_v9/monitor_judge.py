#!/usr/bin/env python3
"""
monitor_judge.py — Monitors ablation judge progress
====================================================
Checks:
  - Process alive?
  - Progress (judgments completed)
  - Stall detection (no new judgments in N minutes)
  - Error rate in log
  - Credit exhaustion (consecutive empty results)
  - Estimated time remaining

Usage:
    python3 eval_v9/monitor_judge.py          # one-shot status
    python3 eval_v9/monitor_judge.py --watch   # continuous monitoring
"""

import json
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

PROJECT = Path(__file__).parent.parent
JUDGMENTS_PATH = PROJECT / "eval_v9" / "results" / "ablation_judgments.jsonl"
LOG_PATH = Path("/tmp/ablation_judge.log")
TOTAL_EXPECTED = 630
STALL_MINUTES = 15


def get_judgments():
    """Load all judgments and return real vs empty counts."""
    if not JUDGMENTS_PATH.exists():
        return [], 0, 0
    records = []
    with open(JUDGMENTS_PATH) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    real = [r for r in records if r.get("total_runs", 0) > 0]
    empty = [r for r in records if r.get("total_runs", 0) == 0]
    return records, len(real), len(empty)


def get_signal_breakdown(records):
    """Break down real judgments by signal."""
    real = [r for r in records if r.get("total_runs", 0) > 0]
    breakdown = {"OPEN": 0, "PAUSE": 0, "WITNESS": 0}
    for r in real:
        sig = r.get("expected_signal", "?")
        if sig in breakdown:
            breakdown[sig] += 1
    return breakdown


def is_process_alive():
    """Check if judge_ablation.py is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "judge_ablation"],
            capture_output=True, text=True
        )
        return result.returncode == 0
    except Exception:
        return False


def get_recent_errors():
    """Count errors in the last 50 lines of log."""
    if not LOG_PATH.exists():
        return 0, []
    try:
        result = subprocess.run(
            ["tail", "-50", str(LOG_PATH)],
            capture_output=True, text=True
        )
        lines = result.stdout.strip().split("\n")
        errors = [l for l in lines if "[error]" in l.lower() or "error" in l.lower()]
        return len(errors), errors[-3:] if errors else []
    except Exception:
        return 0, []


def get_last_judgment_time(records):
    """Get timestamp of most recent real judgment."""
    real = [r for r in records if r.get("total_runs", 0) > 0]
    if not real:
        return None
    timestamps = [r.get("timestamp", "") for r in real]
    timestamps = [t for t in timestamps if t]
    if not timestamps:
        return None
    try:
        return datetime.fromisoformat(max(timestamps))
    except Exception:
        return None


def check_credit_exhaustion(records):
    """Check if recent records have total_runs=0 (credit exhaustion)."""
    if len(records) < 5:
        return False
    last_5 = records[-5:]
    empty = sum(1 for r in last_5 if r.get("total_runs", 0) == 0)
    return empty >= 3


def status_report():
    """Generate a full status report."""
    records, real, empty = get_judgments()
    alive = is_process_alive()
    err_count, recent_errors = get_recent_errors()
    last_time = get_last_judgment_time(records)
    breakdown = get_signal_breakdown(records)
    credit_issue = check_credit_exhaustion(records)

    remaining = TOTAL_EXPECTED - real
    now = datetime.now()

    # Estimate time remaining
    eta_str = "unknown"
    if last_time and real > 233:  # 233 were pre-existing
        new_since_start = real - 233
        if new_since_start > 0:
            # Find the first new judgment timestamp
            new_records = [r for r in records if r.get("total_runs", 0) > 0]
            new_timestamps = sorted([r["timestamp"] for r in new_records])
            if len(new_timestamps) > 233:
                first_new = datetime.fromisoformat(new_timestamps[233])
                elapsed = (now - first_new).total_seconds()
                if elapsed > 0:
                    rate = new_since_start / elapsed  # judgments per second
                    if rate > 0:
                        eta_seconds = remaining / rate
                        eta_str = str(timedelta(seconds=int(eta_seconds)))

    # Stall detection
    stalled = False
    if last_time:
        minutes_since = (now - last_time).total_seconds() / 60
        if minutes_since > STALL_MINUTES and alive:
            stalled = True

    # Report
    print("=" * 55)
    print("  ABLATION JUDGE MONITOR")
    print(f"  {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 55)
    print(f"  Process:    {'RUNNING' if alive else 'STOPPED'}")
    print(f"  Progress:   {real}/{TOTAL_EXPECTED} ({100*real/TOTAL_EXPECTED:.0f}%)")
    print(f"  Remaining:  {remaining}")
    print(f"  ETA:        {eta_str}")
    print(f"  Signals:    OPEN {breakdown['OPEN']}/210 | PAUSE {breakdown['PAUSE']}/210 | WITNESS {breakdown['WITNESS']}/210")
    if last_time:
        ago = (now - last_time).total_seconds() / 60
        print(f"  Last judge: {ago:.1f} min ago")
    if err_count > 0:
        print(f"  Errors:     {err_count} in last 50 log lines")
        for e in recent_errors:
            print(f"              {e.strip()}")
    if empty > 0:
        print(f"  EMPTY:      {empty} records with total_runs=0")

    # Alerts
    alerts = []
    if not alive and real < TOTAL_EXPECTED:
        alerts.append("PROCESS DIED — judge stopped before completion")
    if credit_issue:
        alerts.append("CREDIT EXHAUSTION — recent judgments returning empty")
    if stalled:
        alerts.append(f"STALLED — no new judgment in {STALL_MINUTES}+ minutes")
    if empty > 10:
        alerts.append(f"WARNING — {empty} empty records detected, possible API failures")

    if alerts:
        print(f"\n  {'!' * 50}")
        for a in alerts:
            print(f"  ALERT: {a}")
        print(f"  {'!' * 50}")

    print("=" * 55)

    return {
        "alive": alive,
        "real": real,
        "empty": empty,
        "remaining": remaining,
        "stalled": stalled,
        "credit_issue": credit_issue,
        "alerts": alerts,
    }


if __name__ == "__main__":
    if "--watch" in sys.argv:
        interval = 120  # check every 2 minutes
        print(f"Watching every {interval}s. Ctrl+C to stop.\n")
        while True:
            report = status_report()
            if not report["alive"] and report["remaining"] > 0:
                print("\n  Judge process died. Run with --resume to restart.")
                break
            if report["remaining"] == 0:
                print("\n  ALL JUDGMENTS COMPLETE.")
                break
            print()
            time.sleep(interval)
    else:
        status_report()
