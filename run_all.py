r"""
Full study under the PV protocol, one command.

    python run_all.py --train-root <...\pv_augmented> --pillqc <...\pillQC> --candidate convnext_tiny
    python run_all.py ... --dry-run          # print every command, run nothing
    python run_all.py ... --skip-family      # skip the backbone sweep (stage 1)

Stages. Each finished run is skipped, so after a failure fix the cause and run the
same command again.

  0. Checks     test_metrics.py and verify_amsm_params.py (CPU only). The study stops if either fails.
  1. Family     ConvNeXt candidates at seed 7, for reporting the backbone sweep
  2. Select     the backbone: --candidate if given, otherwise the best VALIDATION F2
                among the EDL seed-7 runs. Test metrics are never read for this.
  3. Train      the six study configurations, on the chosen backbone, at seeds 7, 42, 123
  4. Test       every run: PV test metrics and pillQC transfer (evaluate.py).
                Every EDL run also gets MC-Dropout (T=20).
                The proposed model is compared paired against every other configuration
                at the same seed (exact McNemar and paired bootstrap, Holm-adjusted per seed).
  5. Summarise  mean +/- SD over seeds -> all_results.txt and seeds.csv

Study configurations (variant + head), on the chosen backbone:
  amsm_only + ce       the proposed model: ConvNeXt AMSM under the PV-protocol loss
  amsm_only + edl      evidential head; reported as a comparison row
  amsm_only + cefocal  isolates the focal term
  backbone  + edl      no AMSM: isolates AMSM under the evidential head
  backbone  + ce       no AMSM: isolates AMSM under the PV loss
  amsm_dam  + edl      adds DAM (excluded in config.py; reported as an ablation)

The published PV baselines are not in these tables. Comparisons with them are unpaired
and have to be reported separately, with that caveat.

Nothing here assumes the proposed model wins. The script produces the comparisons, and
the paired tests and seed spread decide what is supported.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from config import PATHS, run_dir_name

SEEDS = (7, 42, 123)
MC_PASSES = 20
PROPOSED = ("amsm_only", "ce")
STUDY = [
    ("amsm_only", "edl"),
    ("amsm_only", "ce"),
    ("amsm_only", "cefocal"),
    ("backbone", "edl"),
    ("backbone", "ce"),
    ("amsm_dam", "edl"),
]
CHECKS = ["test_metrics.py", "verify_amsm_params.py"]


@dataclass(frozen=True)
class Candidate:
    name: str
    backbone: str
    strides: Optional[Tuple[int, ...]] = None
    tag: str = ""
    ce_in_family: bool = True


CANDIDATES = [
    Candidate("convnext_atto", "convnext_atto"),
    Candidate("convnext_femto", "convnext_femto"),
    Candidate("convnext_tiny", "convnext_tiny"),
    Candidate("convnext_small", "convnext_small"),
    Candidate("convnext_base", "convnext_base"),
    Candidate("convnext_tiny_22k", "convnext_tiny_22k"),
    Candidate("convnext_tiny_s4", "convnext_tiny", (4, 8, 16, 32), "s4", ce_in_family=False),
]
BY_NAME = {c.name: c for c in CANDIDATES}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def run_dir(out: Path, c: Candidate, variant: str, head: str, seed: int) -> Path:
    """Same directory name train.py writes (run_dir_name, plus the tag if any)."""
    name = run_dir_name(c.backbone, variant, head, seed)
    return out / (name + (f"_{c.tag}" if c.tag else ""))


def evaluation_complete(path: Path, need) -> bool:
    if not path.exists():
        return False
    rep = json.loads(path.read_text())
    return all(k in rep for k in need)


def run(cmd, dry: bool) -> None:
    print("\n$ " + " ".join(f'"{x}"' if " " in x else x for x in cmd), flush=True)
    if dry:
        return
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        sys.exit(
            f"\nA STEP FAILED (exit code {rc}).\n"
            "Fix the cause and run run_all.py again; finished steps are skipped.\n"
            "Copy the output above and paste it into the chat if you need help."
        )


def train_cmd(a, c: Candidate, variant: str, head: str, seed: int):
    cmd = [sys.executable, "train.py",
           "--train-root", a.train_root,
           "--backbone", c.backbone,
           "--variant", variant,
           "--head", head,
           "--seed", str(seed),
           "--output-dir", str(a.output_dir),
           "--num-workers", str(a.num_workers)]
    if c.strides:
        cmd += ["--strides", *map(str, c.strides)]
    if c.tag:
        cmd += ["--tag", c.tag]
    return cmd


def ensure_trained(a, c: Candidate, variant: str, head: str, seed: int) -> Path:
    d = run_dir(a.output_dir, c, variant, head, seed)
    if (d / "history.json").exists():
        print(f"[skip] {d.name} already trained")
        return d
    run(train_cmd(a, c, variant, head, seed), a.dry_run)
    return d


def val_f2(a, c: Candidate) -> Optional[float]:
    h = run_dir(a.output_dir, c, "amsm_only", "edl", 7) / "history.json"
    if not h.exists():
        return None
    return json.loads(h.read_text())["best_val_f2"]   # validation only


def select(a) -> Optional[Candidate]:
    if a.candidate:
        if a.candidate not in BY_NAME:
            sys.exit(f"Unknown --candidate {a.candidate}. Choices: {', '.join(BY_NAME)}")
        print(f"[select] backbone pinned by --candidate: {a.candidate}")
        return BY_NAME[a.candidate]

    scores = [(val_f2(a, c), c) for c in CANDIDATES]
    missing = [c.name for v, c in scores if v is None]
    if missing:
        if a.dry_run:
            print(f"[select] {len(missing)} EDL runs not trained yet; selection happens "
                  f"after stage 1 when run for real.")
            return None
        sys.exit(f"[select] missing EDL seed-7 runs: {missing}")

    ranked = sorted(scores, key=lambda t: -t[0])
    print("[select] EDL head, seed 7, ranked by VALIDATION F2 (test metrics not read):")
    for v, c in ranked:
        print(f"  {c.name:<20} val F2 {100 * v:7.2f}")
    best = ranked[0][1]
    print(f"[select] backbone = {best.name}")
    return best


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default=PATHS.pv_augmented_root,
                    help="pv_augmented folder (training set augmented offline)")
    ap.add_argument("--pillqc", default=PATHS.pillqc_root,
                    help="pillQC folder for zero-adaptation transfer")
    ap.add_argument("--output-dir", default=PATHS.output_dir)
    ap.add_argument("--num-workers", type=int, default=0, help="0 is safe on Windows")
    ap.add_argument("--candidate", default=None,
                    help="pin the backbone instead of selecting it: " + ", ".join(BY_NAME))
    ap.add_argument("--skip-family", action="store_true",
                    help="skip the backbone sweep in stage 1 (needs --candidate)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command without running it")
    a = ap.parse_args()
    a.output_dir = Path(a.output_dir)

    if a.skip_family and not a.candidate:
        sys.exit("--skip-family needs --candidate, because selection uses the family runs.")
    if not a.dry_run and not Path(a.train_root).exists():
        sys.exit(f"Training data not found: {a.train_root}")
    if not a.dry_run and not Path(a.pillqc).exists():
        print(f"[warn] pillQC folder not found: {a.pillqc}. Transfer results will be "
              f"missing until it is in place and stage 4 is run again.")

    print("=== Stage 0: self-checks ===")
    for script in CHECKS:
        if not Path(script).exists():
            print(f"[warn] {script} not found; skipped")
            continue
        run([sys.executable, script], a.dry_run)

    if not a.skip_family:
        print("\n=== Stage 1: ConvNeXt family (seed 7) ===")
        for c in CANDIDATES:
            ensure_trained(a, c, "amsm_only", "edl", 7)
            if c.ce_in_family:
                ensure_trained(a, c, "amsm_only", "ce", 7)

    print("\n=== Stage 2: backbone ===")
    best = select(a)
    if best is None:   # dry run before any family training
        print("\nStages 3 to 5 depend on the backbone choice and are printed once stage 1 has run.")
        return

    print(f"\n=== Stage 3: study on {best.name} ({len(STUDY)} configurations x {len(SEEDS)} seeds) ===")
    for seed in SEEDS:
        for variant, head in STUDY:
            ensure_trained(a, best, variant, head, seed)

    print(f"\n=== Stage 4: testing on {best.name} ===")
    others = [cfg for cfg in STUDY if cfg != PROPOSED]
    for seed in SEEDS:
        for variant, head in STUDY:
            d = run_dir(a.output_dir, best, variant, head, seed)
            need = ["pillqc"]
            cmd = [sys.executable, "evaluate.py",
                   "--ckpt", str(d / "best.pt"),
                   "--train-root", a.train_root,
                   "--pillqc", a.pillqc,
                   "--num-workers", str(a.num_workers)]
            if head == "edl":
                cmd += ["--mc-dropout", str(MC_PASSES)]
                need.append("mc_dropout")
            if (variant, head) == PROPOSED:
                cmd += ["--compare", *[str(run_dir(a.output_dir, best, v, h, seed)
                                              / "test_predictions.npz") for v, h in others]]
                need.append("comparisons")
            if evaluation_complete(d / "evaluation.json", need):
                print(f"[skip] {d.name} already evaluated")
                continue
            run(cmd, a.dry_run)

    print("\n=== Stage 5: mean +/- SD over seeds ===")
    summary_cmd = [sys.executable, "seed_summary.py", "--runs", str(a.output_dir),
                   "--csv", "seeds.csv"]
    if a.dry_run:
        run(summary_cmd, True)
        return
    res = subprocess.run(summary_cmd, capture_output=True, text=True)
    print(res.stdout + res.stderr)
    header = (
        f"Backbone: {best.name}\n"
        f"Proposed model: {best.name} + {PROPOSED[0]} + {PROPOSED[1]}\n"
        f"Seeds: {', '.join(map(str, SEEDS))}\n"
        "Published PV baselines are not included; comparisons with them are unpaired.\n\n"
    )
    Path("all_results.txt").write_text(header + res.stdout + res.stderr, encoding="utf-8")
    print("[saved] all_results.txt, seeds.csv")
    print(f"\nALL STEPS FINISHED. Backbone: {best.name}")


if __name__ == "__main__":
    main()
