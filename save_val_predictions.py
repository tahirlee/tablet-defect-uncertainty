r"""
Save validation predictions for the three ConvNeXt-Tiny AMSM + CE checkpoints.

Needed so the deferral rate in ensemble_uncertainty.py can be chosen on validation,
not on test. It uses only the validation split and writes val_predictions.npz into
each run folder. Skips any run that already has the file.

    python save_val_predictions.py --train-root <pv_augmented>
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import PATHS, PV, run_dir_name
from data import make_loaders
from evaluate import load_checkpoint
from train import run_inference

SEEDS = (7, 42, 123)


def run_dir(seed: int) -> Path:
    return Path(PATHS.output_dir) / run_dir_name("convnext_tiny", "amsm_only", "ce", seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-root", default=PATHS.pv_augmented_root)
    ap.add_argument("--num-workers", type=int, default=0)
    a = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for seed in SEEDS:
        d = run_dir(seed)
        out = d / "val_predictions.npz"
        if out.exists():
            print(f"[skip] {out} already exists")
            continue
        ckpt = d / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing checkpoint {ckpt}. Run run_all.py first.")

        _, val_loader, _ = make_loaders(a.train_root, PV.batch_size, a.num_workers, seed=seed)
        model = load_checkpoint(str(ckpt), device)
        model.eval()
        pred = run_inference(model, val_loader, device)
        np.savez_compressed(out, **pred)
        print(f"[saved] {out}  ({len(pred['labels'])} validation images)")


if __name__ == "__main__":
    main()
