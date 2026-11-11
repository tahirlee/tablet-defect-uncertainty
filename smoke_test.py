"""
End-to-end smoke test on a synthetic, PV-shaped dataset.

Verifies plumbing, not accuracy: builds 190/190 + 30/30 + 100/100 random
images, runs the offline augmentation to 1140, trains 2 epochs under the real
protocol code, then runs evaluation, pillQC transfer, statistics and the
efficiency profile. Any shape, dtype or API error surfaces here rather than
three hours into a real run.

    python smoke_test.py [--keep]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from config import PV, run_dir_name

HERE = Path(__file__).resolve().parent


def make_fake_pv(root: Path, seed: int = 0) -> None:
    """Random images with a faint class-dependent blob, so learning is possible."""
    rng = np.random.default_rng(seed)
    counts = {
        "train": PV.n_train_per_class,
        "val": PV.n_val_per_class,
        "test": PV.n_test_per_class,
    }
    for split, n in counts.items():
        for label, cls in enumerate(("normal", "defective")):
            d = root / split / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(n):
                arr = rng.integers(90, 170, size=(96, 96, 3), dtype=np.uint8)
                if label == 1:
                    y, x = rng.integers(10, 80, 2)
                    arr[y:y + 12, x:x + 12] = np.clip(
                        arr[y:y + 12, x:x + 12].astype(int) - 55, 0, 255
                    ).astype(np.uint8)
                Image.fromarray(arr).save(d / f"{cls}_{split}_{i:04d}.png")


def make_fake_pillqc(root: Path, seed: int = 1) -> None:
    rng = np.random.default_rng(seed)
    for cls, n in zip(("normal", "chip"), PV.pillqc_expected_counts):
        d = root / "images" / cls
        d.mkdir(parents=True, exist_ok=True)
        for i in range(n):
            arr = rng.integers(70, 190, size=(96, 96, 3), dtype=np.uint8)
            Image.fromarray(arr).save(d / f"{cls}_{i:04d}.png")
    # A dirt class that the loader must ignore.
    d = root / "images" / "dirt"
    d.mkdir(parents=True, exist_ok=True)
    for i in range(5):
        Image.fromarray(
            rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
        ).save(d / f"dirt_{i:04d}.png")


def run(cmd: list, label: str) -> None:
    """Run a step, STREAMING its output line by line.

    Streaming rather than capturing matters here: a training step emits
    nothing for minutes at a time, and a captured subprocess looks
    indistinguishable from a hang. `-u` disables Python's output buffering in
    the child, without which its prints would still arrive in blocks.
    """
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}", flush=True)
    t0 = time.time()
    proc = subprocess.Popen(
        [cmd[0], "-u", *cmd[1:]] if cmd[0] == sys.executable else cmd,
        cwd=HERE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
    )
    tail: deque = deque(maxlen=40)
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        print(line, flush=True)
    code = proc.wait()
    print(f"  [{time.time() - t0:.1f}s]", flush=True)
    if code != 0:
        raise SystemExit(
            f"FAILED: {label} (exit {code})\n"
            + "\n".join(f"  | {t}" for t in tail)
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", action="store_true", help="keep the temp workspace")
    ap.add_argument("--device", default=None,
                    help="default: cuda when available. Using the GPU is both "
                         "faster and a better test, since it exercises the "
                         "same code path real training will use.")
    ap.add_argument("--epochs", type=int, default=2,
                    help="epochs per training step; 1 is enough to smoke-test "
                         "but 2 also checks the scheduler and checkpointing "
                         "across epoch boundaries")
    args = ap.parse_args()

    if args.device is None:
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            args.device = "cpu"
    print(f"[device] {args.device}")
    if args.device == "cpu":
        print("[note] on CPU the four training steps take roughly 10-30 min "
              "in total; output now streams so you can see progress")

    tmp = Path(tempfile.mkdtemp(prefix="pv_smoke_"))
    print(f"[workspace] {tmp}")
    try:
        pv_root = tmp / "PV"
        aug_root = tmp / "PV_aug"
        pillqc = tmp / "pillQC"
        runs = tmp / "runs"

        print("[1/7] synthesizing PV-shaped data ...", flush=True)
        make_fake_pv(pv_root)
        make_fake_pillqc(pillqc)

        # --- module self-tests -------------------------------------------
        run([sys.executable, "backbones.py"], "backbones.py self-test")
        run([sys.executable, "model.py"], "model.py self-test")

        # --- data pipeline ------------------------------------------------
        run([sys.executable, "data.py", "--pv-root", str(pv_root),
             "--out-root", str(aug_root)], "offline augmentation (380 -> 1140)")

        # --- training: both heads, 2 epochs each --------------------------
        common = ["--train-root", str(aug_root), "--epochs", str(args.epochs),
                  "--output-dir", str(runs), "--num-workers", "0",
                  "--no-pretrained", "--device", args.device]
        run([sys.executable, "train.py", *common, "--head", "ce"],
            "train: plain CE (protocol-compliant loss)")
        run([sys.executable, "train.py", *common, "--head", "cefocal"],
            "train: CE + focal (source paper's FairCELoss)")
        run([sys.executable, "train.py", *common, "--head", "edl"],
            "train: evidential head (proposed system)")

        # Names come from config.run_dir_name, the same helper train.py uses to
        # write them, so a change of default backbone cannot desynchronize the
        # two.
        ckpt_edl = runs / run_dir_name(head="edl") / "best.pt"
        preds_ce = runs / run_dir_name(head="ce") / "test_predictions.npz"
        preds_cf = runs / run_dir_name(head="cefocal") / "test_predictions.npz"

        for p in (ckpt_edl, preds_ce, preds_cf):
            if not p.exists():
                raise SystemExit(
                    f"expected artifact missing: {p}\n"
                    f"  training wrote: "
                    f"{[d.name for d in sorted(runs.iterdir())] if runs.exists() else '(nothing)'}\n"
                    f"  config.run_dir_name expects: {ckpt_edl.parent.name}"
                )

        # --- ablation variant ---------------------------------------------
        run([sys.executable, "train.py", *common, "--head", "edl",
             "--variant", "backbone"], "train: backbone-only ablation variant")

        # --- evaluation ----------------------------------------------------
        run([sys.executable, "evaluate.py", "--ckpt", str(ckpt_edl),
             "--train-root", str(aug_root), "--pillqc", str(pillqc),
             "--mc-dropout", "5",
             "--compare", str(preds_ce), str(preds_cf),
             "--num-workers", "0", "--device", args.device],
            "evaluate: in-domain, pillQC transfer, MC-Dropout, paired stats")

        # --- efficiency ----------------------------------------------------
        run([sys.executable, "efficiency.py", "--ckpt", str(ckpt_edl),
             "--runs", "20", "--device", args.device,
             "--out", str(tmp / "efficiency.json")],
            "efficiency: parameters, GFLOPs, latency")

        print("\n" + "=" * 70)
        print("  ALL SMOKE TESTS PASSED")
        print("=" * 70)
        print("\nArtifacts produced:")
        for p in sorted(runs.rglob("*")):
            if p.is_file():
                print(f"  {p.relative_to(runs)}  ({p.stat().st_size:,} bytes)")
    finally:
        if args.keep:
            print(f"\n[kept] {tmp}")
        else:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
