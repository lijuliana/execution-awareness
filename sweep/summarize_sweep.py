#!/usr/bin/env python3
"""Tabulate the sweep: one row per (setting, host). Reads results/<instance>/<job>/result.json, falling
back to the job's log for runs still in progress. usage: python3 summarize_sweep.py [results_dir]"""
import glob
import json
import os
import re
import sys

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
PUBLISHED = {  # both answers right, as trained / after the low-bit flip (sidechan/phenomena/*/report.md, f73d3a7)
    "w-int4": (1.0, 1.0), "w-int8": (0.992, 0.992), "kv-fp8": (1.0, 1.0), "dropout-0.1": (1.0, 1.0),
    "batch": (0.898, 0.0), "tp-4": (0.594, 0.086), "attn-eager": (0.594, 0.047),
}


def gpu(inst):
    p = os.path.join(ROOT, inst, "env", "gpu.csv")
    try:
        return open(p).read().splitlines()[1].split(",")[0].replace("NVIDIA ", "")
    except Exception:
        return "?"


def from_log(path):
    out = {}
    try:
        txt = open(path, errors="replace").read()
    except OSError:
        return out
    pats = {
        "base": r"untrained, end to end on test: (\{.*?\})",
        "trained": r"(?<!un)trained, end to end on test(?: \(condition on blocks below the detector\))?: (\{.*?\})",
        "whole": r"(?<!un)trained, end to end on test \(condition on the whole model\): (\{.*?\})",
        "gauss": r"trained, end to end after Gaussian [^:]*: (\{.*?\})",
        "flip": r"trained, end to end under fresh perturbation \d+: (\{.*?\})",
    }
    for k, pat in pats.items():
        m = re.search(pat, txt)
        if m:
            out[k] = json.loads(m.group(1))
    steps = re.findall(r"step (\d+) loss [\d.]+ dev (\{.*?\})", txt)
    if steps:
        out["dev_last"] = (int(steps[-1][0]), json.loads(steps[-1][1]))
        out["dev_curve"] = [(int(s), json.loads(d)["both_right"]) for s, d in steps]
    m = re.search(r"captured train: (\d+) pairs, (\d+)s", txt)
    if m:
        out["capture_s"] = int(m.group(2))
    m = re.search(r"fingerprint: (\{.*\})", txt)
    if m:
        out["fp"] = json.loads(m.group(1))
    out["crashed"] = "Traceback" in txt
    out["done"] = bool(re.search(r"\] done\s*$", txt, re.M))
    return out


def pct(d, k="both_right"):
    return "  -  " if d is None else f"{100 * d[k]:5.1f}"


rows = []
for inst in sorted(os.listdir(ROOT)):
    logs = glob.glob(os.path.join(ROOT, inst, "logs", "*.log"))
    for lp in sorted(logs):
        job = os.path.basename(lp)[:-4]
        if job in ("queue",) or job.startswith("diag") or job == "rate":
            continue
        r = from_log(lp)
        rows.append((job, gpu(inst), inst, r))

print(f"{'setting':13s} {'GPU':15s} {'untr':>5s} {'dev@last':>10s} {'test':>5s} {'whole':>5s} {'gauss':>5s} {'flip':>5s}  {'published test/flip':>19s}  state")
for job, g, inst, r in sorted(rows):
    dl = r.get("dev_last")
    pub = PUBLISHED.get(job)
    state = "crashed" if r.get("crashed") else "done" if r.get("done") else "running"
    print(
        f"{job:13s} {g:15s} {pct(r.get('base'))} {(f'{100*dl[1]['both_right']:5.1f}@{dl[0]}' if dl else '    -'):>10s}"
        f" {pct(r.get('trained'))} {pct(r.get('whole'))} {pct(r.get('gauss'))} {pct(r.get('flip'))}"
        f"  {(f'{100*pub[0]:.0f} / {100*pub[1]:.0f}' if pub else '-'):>19s}  {state} ({inst})"
    )
print("\nAll numbers: percent of held-out prompts with both answers right (Yes with the setting on, No with it off).")
