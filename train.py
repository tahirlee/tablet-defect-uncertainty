"""
Train UAMSF-Net under the PharmaVision protocol.

  Adam, learning rate 1e-4, weight decay 1e-4, batch size 16
  linear warm-up over the first epoch, then cosine decay to 1% of the learning rate
  at most 100 epochs; the checkpoint with the best validation F2-score
  (decision threshold 0.40) is kept; early stopping after 15 epochs without improvement
  cross-entropy loss, all layers fine-tuned

    python train.py --data data/PV_augmented --seed 7
    python train.py --data data/PV_augmented --seed 42
    python train.py --data data/PV_augmented --seed 123

Each run writes runs/seed<k>/best.pt and runs/seed<k>/history.json.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR

from data import list_pv, make_loader, seed_everything
from model import UAMSFNet

LR, WEIGHT_DECAY, BATCH_SIZE = 1e-4, 1e-4, 16
MAX_EPOCHS, WARMUP_EPOCHS, FINAL_LR_FRACTION, PATIENCE = 100, 1, 0.01, 15
THRESHOLD = 0.40


def f2_score(labels: np.ndarray, probs: np.ndarray, threshold: float = THRESHOLD) -> float:
    pred = (probs >= threshold).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    return 5 * tp / (5 * tp + 4 * fn + fp) if tp else 0.0


@torch.no_grad()
def predict(model: nn.Module, loader, device) -> dict:
    """P(defective), labels and image paths for every image in the loader."""
    model.eval()
    probs, labels, paths = [], [], []
    for x, y, p in loader:
        probs.append(model(x.to(device))["probs"][:, 1].cpu().numpy())
        labels.append(np.asarray(y))
        paths.extend(p)
    return {"scores": np.concatenate(probs), "labels": np.concatenate(labels).astype(int),
            "paths": np.array(paths)}


def scheduler(optimizer, steps_per_epoch: int, max_epochs: int) -> LambdaLR:
    warmup = max(1, steps_per_epoch * WARMUP_EPOCHS)
    total = max(warmup + 1, steps_per_epoch * max_epochs)

    def factor(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, factor)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/PV_augmented")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    seed_everything(a.seed)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    pv = list_pv(a.data)
    train_loader = make_loader(pv["train"], BATCH_SIZE, shuffle=True, seed=a.seed,
                               num_workers=a.workers)
    val_loader = make_loader(pv["val"], BATCH_SIZE, num_workers=a.workers)
    print(f"[data] train {len(pv['train'])}, validation {len(pv['val'])}")

    model = UAMSFNet(pretrained=True).to(device)
    print(f"[model] {model.param_counts()['total']:,} parameters")
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = scheduler(optimizer, len(train_loader), a.epochs)

    run_dir = Path(a.out) / f"seed{a.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    best_f2, best_epoch, wait, history = -1.0, -1, 0, []
    t0 = time.time()

    for epoch in range(a.epochs):
        model.train()
        total, n = 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(x)["logits"], y)
            loss.backward()
            optimizer.step()
            sched.step()
            total, n = total + float(loss.detach()), n + 1

        val = predict(model, val_loader, device)
        f2 = f2_score(val["labels"], val["scores"])
        history.append({"epoch": epoch, "train_loss": total / max(n, 1), "val_f2": f2})
        improved = f2 > best_f2
        if improved:
            best_f2, best_epoch, wait = f2, epoch, 0
            torch.save({"model": model.state_dict(), "epoch": epoch, "val_f2": f2},
                       run_dir / "best.pt")
        else:
            wait += 1
        print(f"epoch {epoch:3d}  loss {total / max(n, 1):.4f}  val F2 {f2:.4f}"
              f"{'  *' if improved else ''}")
        if wait >= PATIENCE:
            print(f"[stop] no improvement for {PATIENCE} epochs")
            break

    (run_dir / "history.json").write_text(json.dumps(
        {"best_epoch": best_epoch, "best_val_f2": best_f2, "seed": a.seed,
         "minutes": (time.time() - t0) / 60, "history": history}, indent=2))
    print(f"[done] best validation F2 {best_f2:.4f} at epoch {best_epoch}; saved {run_dir}")


if __name__ == "__main__":
    main()
