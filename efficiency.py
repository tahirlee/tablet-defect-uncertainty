"""
Computational profiling, matching the PV protocol (Sect. 2.4).

  parameters  -- total trainable, in millions
  GFLOPs      -- one forward pass at 224 x 224
  latency     -- MEDIAN wall-clock ms for a single image (batch size 1), over
                 500 timed runs after 50 warm-up runs, GPU-synchronized around
                 each pass; variability reported as the interquartile range
  throughput  -- 1000 / median latency, in FPS

The PV paper's own finding is worth keeping in view: parameter count and
GFLOPs were unreliable predictors of measured latency, because densely
connected and heavily block-structured operations parallelize poorly on GPU
regardless of their theoretical cost. Quote the measured latency, not GFLOPs,
when arguing a model can keep pace with an inspection line -- and note that
single-image latency is not end-to-end pipeline throughput, which also depends
on acquisition, transfer and scheduling.

Usage
    python efficiency.py --backbones convnext_tiny efficientnetv2_s resnet50
    python efficiency.py --ckpt runs/convnext_tiny_amsm_only_edl_seed7/best.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from backbones import available_backbones
from config import PV
from model import build_model


def count_flops(model: torch.nn.Module, device, size: int = PV.image_size) -> Optional[float]:
    """GFLOPs for one forward pass. Returns None if no counter is installed."""
    x = torch.randn(1, 3, size, size, device=device)
    try:
        from fvcore.nn import FlopCountAnalysis

        fca = FlopCountAnalysis(model, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return float(fca.total()) / 1e9
    except Exception:
        pass
    try:
        from thop import profile

        macs, _ = profile(model, inputs=(x,), verbose=False)
        return float(macs) * 2 / 1e9   # MACs -> FLOPs
    except Exception:
        print("[warn] install fvcore or thop to report GFLOPs")
        return None


@torch.no_grad()
def measure_latency(
    model: torch.nn.Module,
    device,
    size: int = PV.image_size,
    warmup: int = PV.latency_warmup_runs,
    runs: int = PV.latency_timed_runs,
    batch_size: int = 1,
) -> Dict[str, float]:
    model.eval()
    x = torch.randn(batch_size, 3, size, size, device=device)
    cuda = device.type == "cuda"

    for _ in range(warmup):
        model(x)
    if cuda:
        torch.cuda.synchronize()

    timings: List[float] = []
    for _ in range(runs):
        if cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            model(x)
            end.record()
            torch.cuda.synchronize()
            timings.append(start.elapsed_time(end))
        else:
            import time

            t0 = time.perf_counter()
            model(x)
            timings.append((time.perf_counter() - t0) * 1000.0)

    arr = np.asarray(timings)
    median = float(np.median(arr))
    return {
        "median_latency_ms": median,
        "iqr_ms": float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        "mean_latency_ms": float(arr.mean()),
        "throughput_fps": 1000.0 / median if median > 0 else float("nan"),
        "batch_size": batch_size,
        "runs": runs,
    }


def peak_memory_mb(model: torch.nn.Module, device, size: int = PV.image_size) -> Optional[float]:
    if device.type != "cuda":
        return None
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        model(torch.randn(1, 3, size, size, device=device))
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def profile_model(model: torch.nn.Module, device, label: str) -> dict:
    model = model.to(device).eval()
    counts = model.param_counts()
    row = {
        "model": label,
        "params_m": counts["total"] / 1e6,
        "backbone_params_m": counts["backbone"] / 1e6,
        "amsm_params_m": counts["amsm"] / 1e6,
        "gflops": count_flops(model, device),
        **measure_latency(model, device),
        "peak_memory_mb": peak_memory_mb(model, device),
    }
    return row


def print_table(rows: List[dict]) -> None:
    hdr = (f"{'model':<34}{'params(M)':>11}{'GFLOPs':>9}"
           f"{'median(ms)':>12}{'IQR':>8}{'FPS':>9}")
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        g = f"{r['gflops']:.2f}" if r["gflops"] is not None else "n/a"
        print(f"{r['model']:<34}{r['params_m']:>11.2f}{g:>9}"
              f"{r['median_latency_ms']:>12.2f}{r['iqr_ms']:>8.2f}"
              f"{r['throughput_fps']:>9.1f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="profile a trained checkpoint")
    ap.add_argument("--backbones", nargs="*", default=None,
                    help="profile untrained models across these backbones")
    ap.add_argument("--variants", nargs="*", default=["amsm_only"])
    ap.add_argument("--batch-sizes", type=int, nargs="*", default=[1])
    ap.add_argument("--runs", type=int, default=PV.latency_timed_runs)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="efficiency.json")
    args = ap.parse_args()

    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[device] {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))
    if device.type != "cuda":
        print("[warn] CPU timings are not comparable to the paper's GPU figures.")

    rows: List[dict] = []
    if args.ckpt:
        from evaluate import load_checkpoint

        model = load_checkpoint(args.ckpt, device)
        rows.append(profile_model(model, device, Path(args.ckpt).parent.name))
    else:
        for bb in (args.backbones or [available_backbones()[0]]):
            for v in args.variants:
                model = build_model(variant=v, backbone=bb, pretrained=False)
                rows.append(profile_model(model, device, f"{bb} / {v}"))

    print_table(rows)

    if len(args.batch_sizes) > 1 and rows:
        print("\nlatency by batch size:")
        model = model.to(device).eval()
        for bs in args.batch_sizes:
            lat = measure_latency(model, device, runs=max(50, args.runs // 5), batch_size=bs)
            per_img = lat["median_latency_ms"] / bs
            print(f"  bs={bs:<4} median {lat['median_latency_ms']:7.2f} ms "
                  f"({per_img:.2f} ms/image)")

    with open(args.out, "w") as fh:
        json.dump(rows, fh, indent=2, default=float)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
