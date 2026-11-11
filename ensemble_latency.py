r"""
Latency of the 3-seed ensemble under the PV protocol, with the same settings as efficiency.py:
batch 1, 224 x 224, 50 warm-up runs, 500 timed runs, GPU-synchronized around each pass.

The ensemble is three ConvNeXt-Tiny AMSM + CE models run one after another and averaged.
One CE model is profiled on its own for comparison.

    python ensemble_latency.py

Writes ensemble_latency.json and prints a table.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from config import PATHS, run_dir_name
from efficiency import measure_latency, peak_memory_mb
from evaluate import load_checkpoint

SEEDS = (7, 42, 123)
OUT = "ensemble_latency.json"


def run_dir(seed: int) -> Path:
    return Path(PATHS.output_dir) / run_dir_name("convnext_tiny", "amsm_only", "ce", seed)


class Ensemble(torch.nn.Module):
    """Average of the defective probabilities of several models."""

    def __init__(self, models):
        super().__init__()
        self.models = torch.nn.ModuleList(models)

    def forward(self, x):
        probs = [torch.softmax(m(x)["logits"], dim=1)[:, 1] for m in self.models]
        return torch.stack(probs).mean(dim=0)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("[warn] CPU timings are not comparable to the paper's GPU figures.")

    models = []
    for seed in SEEDS:
        ckpt = run_dir(seed) / "best.pt"
        if not ckpt.exists():
            raise SystemExit(f"Missing {ckpt}. Run run_all.py first.")
        models.append(load_checkpoint(str(ckpt), device).eval())

    rows = []

    single = models[0]
    lat = measure_latency(single, device)
    rows.append({
        "model": "ConvNeXt-Tiny AMSM + CE (one seed)",
        "params_m": sum(p.numel() for p in single.parameters()) / 1e6,
        **lat,
        "peak_memory_mb": peak_memory_mb(single, device),
    })

    ens = Ensemble(models).to(device).eval()
    lat = measure_latency(ens, device)
    rows.append({
        "model": "ConvNeXt-Tiny AMSM + CE (3-seed ensemble)",
        "params_m": sum(p.numel() for p in ens.parameters()) / 1e6,
        **lat,
        "peak_memory_mb": peak_memory_mb(ens, device),
    })

    hdr = f"{'model':<44}{'params(M)':>11}{'median(ms)':>12}{'IQR':>8}{'FPS':>9}{'peak MB':>10}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        mem = f"{r['peak_memory_mb']:.0f}" if r["peak_memory_mb"] is not None else "n/a"
        print(f"{r['model']:<44}{r['params_m']:>11.2f}{r['median_latency_ms']:>12.2f}"
              f"{r['iqr_ms']:>8.2f}{r['throughput_fps']:>9.1f}{mem:>10}")

    Path(OUT).write_text(json.dumps(rows, indent=2, default=float), encoding="utf-8")
    print(f"\n[saved] {OUT}")


if __name__ == "__main__":
    main()
