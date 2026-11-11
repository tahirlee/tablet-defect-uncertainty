"""
Mean and standard deviation across random seeds, one row per configuration.

Runs are grouped by their directory name with the `_seedN` part removed, so
convnext_tiny_amsm_only_edl_seed7, ..._seed42 and ..._seed123 form one group.
Validation F2 is the selection metric; test numbers are reported, not used to
choose anything.

Torch-free: reads history.json only.

    python seed_summary.py                 # reads ./runs
    python seed_summary.py --runs runs --csv seeds.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

SEED_PART = re.compile(r"_seed\d+")
METRICS = [
    ("val_f2", "val F2"),
    ("test_acc", "test acc"),
    ("test_prec", "prec"),
    ("test_rec", "recall"),
    ("test_f1", "test F1"),
    ("test_f2", "test F2"),
]


def load(run_dir: Path):
    hist = run_dir / "history.json"
    if not hist.exists():
        return None
    h = json.loads(hist.read_text())
    seed = h.get("protocol", {}).get("seed")
    t = h["test_metrics"]
    return {
        "group": SEED_PART.sub("", run_dir.name),
        "seed": seed,
        "val_f2": h["best_val_f2"],
        "test_acc": t["accuracy"],
        "test_prec": t["precision"],
        "test_rec": t["recall"],
        "test_f1": t["f1"],
        "test_f2": t["f2"],
    }


def spread(values):
    mean = statistics.fmean(values)
    sd = statistics.stdev(values) if len(values) > 1 else float("nan")
    return mean, sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    groups = defaultdict(list)
    for d in sorted(Path(args.runs).iterdir()):
        if d.is_dir():
            r = load(d)
            if r is not None:
                groups[r["group"]].append(r)
    if not groups:
        print(f"No finished runs found in {args.runs}")
        return

    out_rows = []
    print("Values are mean ± sample standard deviation over seeds, in percent.\n")
    for group, runs in sorted(groups.items()):
        seeds = sorted(r["seed"] for r in runs if r["seed"] is not None)
        print(f"{group}   seeds {seeds}   n = {len(runs)}")
        if len(runs) < 3:
            print("   WARNING: fewer than 3 seeds; the spread is not reliable.")
        row = {"group": group, "n": len(runs), "seeds": " ".join(map(str, seeds))}
        for key, label in METRICS:
            mean, sd = spread([r[key] for r in runs])
            print(f"   {label:<10} {100 * mean:7.2f} ± {100 * sd:5.2f}")
            row[f"{key}_mean"] = 100 * mean
            row[f"{key}_sd"] = 100 * sd
        out_rows.append(row)
        print()

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
            w.writeheader()
            w.writerows(out_rows)
        print(f"[saved] {args.csv}")


if __name__ == "__main__":
    main()
