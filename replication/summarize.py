#!/usr/bin/env python3
"""Summarize the L40S replication bundle against the values Sauers reported."""
import json, sys
from pathlib import Path

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out")
EXPECTED = {  # from results/gpu-transfer-062 (L40S row) and compiler-control-048
    "trained_reference": {"accuracy": 1.0, "shift": 27.25},
    "base_reference": {"accuracy": 0.5, "shift": -0.053},
    "compile_control_prompts_differing": 1,
}

def events(path):
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try: out.append(json.loads(line))
            except json.JSONDecodeError: pass
    return out

def stage_results(dirname):
    rows = {}
    for e in events(root / dirname / "metrics.jsonl"):
        if e.get("event") == "stage_result":
            s = e["summary"]
            rows[f'{e["stage"]}_{e["comparison"]}'] = {
                "accuracy": s["accuracy"], "both_correct": s["both_conditions_correct"],
                "shift": s["paired_log_odds_shift"], "shift_ci95": s["paired_log_odds_shift_ci95"],
                "pos/zero/neg": (s["positive_shifts"], s["zero_shifts"], s["negative_shifts"]),
                "mean_ce": s["mean_ce"],
            }
    parity = [e for e in events(root / dirname / "metrics.jsonl") if e.get("event") == "buffer_parity"]
    return rows, parity

print(f"# Replication summary: {root}\n")
env = root / "env"
for f in ["gpu.csv", "torch.txt", "source_commit.txt", "artifact_sha.txt", "data_sha.txt"]:
    p = env / f
    if p.exists():
        print(f"{f}: " + " | ".join(l for l in p.read_text().splitlines() if l.strip()))
print()
status = events(root / "stage_status.jsonl")
print("stage exits:", {s["stage"]: s["exit"] for s in status})
print()

for label, dirname in [("A. Released 034 detector, fresh process", "A_fresh034"),
                       ("E. Retrained detector, fresh process", "E_fresh_retrain")]:
    rows, parity = stage_results(dirname)
    print(f"## {label}")
    if not rows:
        print("  (no stage_result events)\n"); continue
    print(f"  {'cell':22s} {'acc':>6s} {'both':>6s} {'shift (nats)':>14s} {'95% CI':>22s} {'+/0/-':>12s}")
    for k, r in rows.items():
        ci = f"[{r['shift_ci95'][0]:+.2f}, {r['shift_ci95'][1]:+.2f}]"
        print(f"  {k:22s} {r['accuracy']:6.3f} {r['both_correct']:6.3f} {r['shift']:+14.3f} {ci:>22s} {str(r['pos/zero/neg']):>12s}")
    exp = EXPECTED["trained_reference"]
    if "trained_reference" in rows:
        r = rows["trained_reference"]
        verdict = "REPLICATES" if r["accuracy"] == exp["accuracy"] and abs(r["shift"] - exp["shift"]) < 1.0 else "DIFFERS"
        print(f"  expected trained_reference: acc {exp['accuracy']}, shift ~{exp['shift']} nats  ->  {verdict}")
    if parity:
        bit = sum(1 for p in parity if p["bit_identical"])
        print(f"  buffer-copy parity: {bit}/{len(parity)} comparisons bit-identical (Sauers: 0, the buffer probe changes logits)")
    print()

print("## B. Compile-to-compile control (two independent compilations of the unprobed model)")
summ = root / "B_compile_control" / "summary.json"
if summ.exists():
    s = json.loads(summ.read_text())
    ab = s["comparisons"].get("none_a__none_b", [])
    diff = [i for i, c in enumerate(ab) if not c["bit_identical"]]
    print(f"  prompts differing: {len(diff)}/{len(ab)} (Sauers run 048: 1/4)")
    for i, c in enumerate(ab):
        print(f"  prompt {i}: bit_identical={c['bit_identical']} fraction_different={c['fraction_different']:.4f} max_abs={c['max_abs_difference']}")
    for k in ["interleaved_repeat_parity", "final_wrapper_identity_verified", "post_warmup_compile_counters_unchanged", "complete"]:
        print(f"  {k}: {s.get(k)}")
else:
    print("  (no summary.json)")
print()

print("## C. Retrained 034 recipe (256 prompts, 1024 updates)")
fit = root / "C_retrain" / "fit"
ev = events(fit / "metrics.jsonl")
kinds = {}
for e in ev:
    kinds[e.get("event")] = kinds.get(e.get("event"), 0) + 1
print("  event counts:", kinds)
evals = [e for e in ev if e.get("event") in ("evaluation", "eval", "report", "cohort_result", "stage_result")]
for e in evals[-8:]:
    print("  " + json.dumps(e)[:400])
for name in sorted(fit.glob("*split_inference.json")):
    try:
        d = json.loads(name.read_text())
        summ = d.get("summary", d if isinstance(d, dict) else {})
        acc = summ.get("accuracy"); shift = summ.get("paired_log_odds_shift")
        print(f"  {name.name}: accuracy={acc} shift={shift}")
    except Exception as ex:
        print(f"  {name.name}: unreadable ({ex})")
print()

print("## D. Identical-input null on the retrain (expected: accuracy 0.5, shift exactly 0)")
null = root / "D_null"
ev = events(null / "metrics.jsonl") or events(null / "fit" / "metrics.jsonl")
for e in [x for x in ev if x.get("event") not in ("prediction", "progress")][-6:]:
    print("  " + json.dumps(e)[:400])
for name in sorted(null.rglob("*split_inference.json"))[-3:]:
    try:
        d = json.loads(name.read_text()); summ = d.get("summary", {})
        print(f"  {name.relative_to(root)}: accuracy={summ.get('accuracy')} shift={summ.get('paired_log_odds_shift')}")
    except Exception as ex:
        print(f"  {name.name}: unreadable ({ex})")
