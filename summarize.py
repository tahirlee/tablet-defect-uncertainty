"""
Summarize finished training runs from runs/*/history.json.

Runs are RANKED BY VALIDATION F2, the protocol's selection metric. Test columns
are printed for reporting only. Choosing a model by its test numbers would
contaminate the comparison, so this script never sorts by test.

Torch-free: it only reads JSON, so it runs on any machine.

    python summarize.py                       # reads ./runs
    python summarize.py --runs runs --csv family.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load(run_dir: Path) -> dict | None:
    hist = run_dir / "history.json"
    if not hist.exists():
        return None   # interrupted or unfinished run; ignored, not an error
    h = json.loads(hist.read_text())
    t = h["test_metrics"]
    return {
        "run": run_dir.name,
        "params_M": h["param_counts"]["total"] / 1e6,
        "best_epoch": h["best_epoch"],
        "val_f2": h["best_val_f2"],
        "test_acc": t["accuracy"],
        "test_prec": t["precision"],
        "test_rec": t["recall"],
        "test_f1": t["f1"],
        "test_f2": t["f2"],
        "minutes": h.get("train_minutes", float("nan")),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", default=None, help="also write the table to this CSV file")
    args = ap.parse_args()

    rows = [r for r in (load(d) for d in sorted(Path(args.runs).iterdir())
                        if d.is_dir()) if r is not None]
    if not rows:
        print(f"No finished runs found in {args.runs}")
        return
    rows.sort(key=lambda r: r["val_f2"], reverse=True)

    hdr = (f"{'run':<42}{'params':>8}{'ep*':>5}{'valF2':>8}"
           f"{'testAcc':>9}{'prec':>7}{'rec':>7}{'testF1':>8}{'min':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['run']:<42}{r['params_M']:>7.2f}M{r['best_epoch']:>5}"
              f"{100 * r['val_f2']:>8.2f}{100 * r['test_acc']:>9.2f}"
              f"{100 * r['test_prec']:>7.2f}{100 * r['test_rec']:>7.2f}"
              f"{100 * r['test_f1']:>8.2f}{r['minutes']:>7.1f}")
    print("-" * len(hdr))
    print("ep* = epoch of the saved checkpoint. valF2 is the selection metric;\n"
          "test columns are for reporting only. Values are percentages.")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[saved] {args.csv}")


if __name__ == "__main__":
    main()
