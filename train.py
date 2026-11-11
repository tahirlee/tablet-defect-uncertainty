"""
Training, under the PharmaVision protocol exactly.

  Adam (not AdamW), lr 1e-4, weight decay 1e-4, batch 16, max 100 epochs
  linear warm-up over epoch 1, then cosine decay to 1% of the base lr
  checkpoint on best VALIDATION F2-score at a 0.40 decision threshold
  early stopping after 15 epochs without F2 improvement
  seed 7, single run

Usage
    python train.py --train-root data/PV_augmented --variant amsm_only --head edl
    python train.py --train-root data/PV_augmented --variant amsm_only --head ce
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR

from backbones import available_backbones
from config import MODEL, PV, PATHS, run_dir_name
from data import make_loaders, seed_everything
from losses import LOSSES, build_loss
from metrics import compute_metrics
from model import build_model, VARIANTS


# ---------------------------------------------------------------------------
# Schedule
# ---------------------------------------------------------------------------
def build_scheduler(optimizer, steps_per_epoch: int, max_epochs: int) -> LambdaLR:
    """Linear warm-up for one epoch, then cosine decay to cosine_final_lr_frac."""
    warmup_steps = max(1, steps_per_epoch * PV.warmup_epochs)
    total_steps = max(warmup_steps + 1, steps_per_epoch * max_epochs)
    floor = PV.cosine_final_lr_frac

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, fn)


# ---------------------------------------------------------------------------
# Inference helper -- shared with evaluate.py
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_inference(model, loader, device) -> Dict[str, np.ndarray]:
    """Single deterministic forward pass over a loader.

    Returns P(defective), full probability vectors, the scalar uncertainty,
    labels, AMSM gate weights and file paths.
    """
    model.eval()
    probs, uncert, labels, paths, gates = [], [], [], [], []

    for x, y, p in loader:
        out = model(x.to(device, non_blocking=True))
        probs.append(out["probs"].cpu().numpy())
        uncert.append(out["uncertainty"].cpu().numpy())
        if "gates" in out:
            gates.append(out["gates"].cpu().numpy())
        labels.append(np.asarray(y))
        paths.extend(p)

    result = {
        "probs": np.concatenate(probs),
        "scores": np.concatenate(probs)[:, 1],   # P(defective)
        "uncertainty": np.concatenate(uncert),
        "labels": np.concatenate(labels),
        "paths": np.array(paths),
    }
    if gates:
        result["gates"] = np.concatenate(gates)  # [N, n_stages, 4]
    return result


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(args) -> Path:
    seed_everything(args.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))

    train_loader, val_loader, test_loader = make_loaders(
        args.train_root, PV.batch_size, args.num_workers, seed=args.seed
    )
    print(f"[data] train {len(train_loader.dataset)} | "
          f"val {len(val_loader.dataset)} | test {len(test_loader.dataset)}")

    model = build_model(
        variant=args.variant,
        head_type=args.head,
        backbone=args.backbone,
        strides=tuple(args.strides),
        proj_channels=(None if args.proj_channels <= 0 else args.proj_channels),
        amsm_norm=args.amsm_norm,
        num_classes=2,
        pretrained=not args.no_pretrained,
        max_parameters=(None if args.max_params <= 0 else args.max_params),
    ).to(device)
    counts = model.param_counts()
    print("[model]\n" + model.describe())

    criterion = build_loss(args.head, 2).to(device)
    # All layers fine-tuned; nothing frozen (PV Sect. 2.3).
    optimizer = torch.optim.Adam(
        model.parameters(), lr=PV.lr, weight_decay=PV.weight_decay
    )
    scheduler = build_scheduler(optimizer, len(train_loader), args.epochs)

    # --tag keeps variants that share a backbone/variant/head from overwriting
    # each other (e.g. a stride-4 run must not replace the default run).
    run_dir = Path(args.output_dir) / (
        run_dir_name(args.backbone, args.variant, args.head, args.seed)
        + (f"_{args.tag}" if args.tag else "")
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = run_dir / "best.pt"

    best_f2, best_epoch, bad_epochs = -1.0, -1, 0
    history: List[dict] = []
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            parts = criterion(out, y, epoch=epoch)
            parts["loss"].backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()

            running += float(parts["loss"].detach())
            n_batches += 1

        val = run_inference(model, val_loader, device)
        vm = compute_metrics(val["labels"], val["scores"], PV.decision_threshold)

        row = {
            "epoch": epoch,
            "train_loss": running / max(n_batches, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "val_f2": vm.f2, "val_f1": vm.f1,
            "val_accuracy": vm.accuracy, "val_recall": vm.recall,
            "val_precision": vm.precision,
        }
        history.append(row)

        improved = vm.f2 > best_f2
        if improved:
            best_f2, best_epoch, bad_epochs = vm.f2, epoch, 0
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "val_f2": vm.f2,
                "variant": args.variant,
                "head": args.head,
                "backbone": args.backbone,
                "strides": tuple(args.strides),
                "proj_channels": (None if args.proj_channels <= 0 else args.proj_channels),
                "amsm_norm": model.amsm_norm,
            }, ckpt_path)
        else:
            bad_epochs += 1

        print(f"ep {epoch:3d}  loss {row['train_loss']:.4f}  "
              f"val F2 {vm.f2:.4f}  F1 {vm.f1:.4f}  acc {vm.accuracy:.4f}"
              f"{'  *' if improved else ''}")

        if bad_epochs >= PV.early_stopping_patience:
            print(f"[stop] no val F2 improvement for "
                  f"{PV.early_stopping_patience} epochs")
            break

    elapsed = time.time() - t0
    print(f"[done] best val F2 {best_f2:.4f} at epoch {best_epoch} "
          f"({elapsed / 60:.1f} min)")

    # Reload the selected checkpoint and freeze its test predictions.
    model.load_state_dict(torch.load(ckpt_path, map_location=device)["model"])
    test = run_inference(model, test_loader, device)
    tm = compute_metrics(test["labels"], test["scores"], PV.decision_threshold)
    print(f"[test] {tm.pretty()}")

    np.savez_compressed(
        run_dir / "test_predictions.npz",
        **{k: v for k, v in test.items()},
    )
    with open(run_dir / "history.json", "w") as fh:
        json.dump({
            "history": history,
            "best_epoch": best_epoch,
            "best_val_f2": best_f2,
            "test_metrics": tm.as_dict(),
            "param_counts": counts,
            "train_minutes": elapsed / 60,
            "protocol": {
                "lr": PV.lr, "weight_decay": PV.weight_decay,
                "batch_size": PV.batch_size, "max_epochs": args.epochs,
                "threshold": PV.decision_threshold,
                "selection_metric": PV.selection_metric,
                "patience": PV.early_stopping_patience, "seed": args.seed,
                "loss": args.head,
                "grad_clip": args.grad_clip,
            },
        }, fh, indent=2)

    print(f"[saved] {run_dir}")
    return run_dir


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", default=PATHS.pv_augmented_root,
                    help="PV root containing train/val/test (augmented training set)")
    ap.add_argument("--variant", default="amsm_only", choices=list(VARIANTS))
    ap.add_argument("--head", default="edl", choices=list(LOSSES),
                    help="; ".join(f"{k}: {v}" for k, v in LOSSES.items()))
    ap.add_argument("--grad-clip", type=float, default=0.0,
                    help="max grad norm; 0 disables. OFF by default because the "
                         "PV protocol does not specify clipping -- enabling it "
                         "for one run only is an unreported advantage. If the "
                         "evidential run diverges, enable it for EVERY run and "
                         "say so.")
    ap.add_argument("--backbone", default=MODEL.backbone, choices=available_backbones(),
                    help=f"ImageNet-pretrained encoder (default {MODEL.backbone})")
    ap.add_argument("--strides", type=int, nargs="+", default=list(MODEL.feature_strides),
                    help="feature taps by stride, e.g. --strides 4 8 16 32 for small defects")
    ap.add_argument("--proj-channels", type=int, default=MODEL.proj_channels or 0,
                    help="cap each stage at this width before AMSM; 0 disables")
    ap.add_argument("--amsm-norm", default=None, choices=[None, "bn", "gn"],
                    help="default follows the backbone (gn for LayerNorm backbones)")
    ap.add_argument("--epochs", type=int, default=PV.max_epochs)
    ap.add_argument("--output-dir", default=PATHS.output_dir)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=PV.seed,
                    help="random seed for initialisation, shuffling and dropout. "
                         "7 is the PV protocol seed; 42 and 123 are additional "
                         "seeds for variance reporting.")
    ap.add_argument("--device", default=None)
    ap.add_argument("--no-pretrained", action="store_true")
    ap.add_argument("--tag", default="",
                    help="suffix appended to the run directory name, for variants "
                         "such as a different --strides")
    ap.add_argument("--max-params", type=int, default=(MODEL.max_parameters or 0),
                    help="parameter ceiling checked at model construction; 0 disables "
                         "the check. Default is the 5M cap from config.py.")
    train(ap.parse_args())


if __name__ == "__main__":
    main()
