"""
Parameter budget explorer.

Builds the full model across candidate backbones and projection widths and
prints the breakdown against a cap, next to PV's benchmarked architectures so
the comparison is concrete.

Backbones are constructed with pretrained=False here -- weights do not change
parameter counts, and this keeps the script fast and offline.

    python param_budget.py                      # default sweep, 5M cap
    python param_budget.py --cap 3000000
    python param_budget.py --backbones efficientnet_b0 mobilenet_v3_large \
                           --proj 64 96 128 192 256
"""
from __future__ import annotations

import argparse
from typing import List, Optional

from backbones import REGISTRY, available_backbones
from config import MODEL

# PV Table 11, for context. Parameter counts in millions.
PV_BASELINES = [
    ("ShuffleNetV2", 1.26, 84.78, 49.44),
    ("MobileNetV3-small", 1.52, 71.28, 44.83),
    ("GoogLeNet", 5.60, 83.06, 76.19),
    ("DenseNet-169", 12.48, 78.45, 48.39),
    ("DLA-34", 15.27, 80.81, 63.89),
    ("EfficientNetV2-S", 20.18, 88.56, 75.00),
    ("Xception", 20.81, 87.08, 70.27),
    ("ResNet-50", 23.51, 83.84, 72.46),
    ("AlexNet", 57.01, 67.91, 29.06),
    ("Swin-B", 86.74, 85.71, 40.62),
    ("ViT-B/32", 87.46, 81.82, 38.89),
    ("VGG-19", 139.58, 84.51, 72.94),
]

DEFAULT_BACKBONES = [
    "mobilenet_v2", "mobilenet_v3_large", "convnext_atto",
    "regnet_y_400mf", "efficientnet_b0", "convnext_femto",
]


def build_row(backbone: str, proj: Optional[int], cap: Optional[int]) -> Optional[dict]:
    from model import build_model

    try:
        m = build_model(
            variant="amsm_only", head_type="edl", backbone=backbone,
            proj_channels=proj, pretrained=False, max_parameters=None,
        )
    except Exception as exc:
        return {"backbone": backbone, "proj": proj, "error": f"{type(exc).__name__}: {exc}"}

    c = m.param_counts()
    # NOTE: param_counts() also has a 'backbone' key (its parameter count), so
    # the model NAME is stored under 'name' to avoid being overwritten by **c.
    return {
        "name": backbone,
        "proj": proj,
        "stage_in": m.backbone.out_channels,
        "stage_out": m.stage_channels,
        "feat_dim": m.feat_dim,
        **c,
        "within_cap": (cap is None or c["total"] <= cap),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cap", type=int, default=5_000_000)
    ap.add_argument("--backbones", nargs="*", default=DEFAULT_BACKBONES)
    ap.add_argument("--proj", type=int, nargs="*", default=[96, 128, 192, 256])
    ap.add_argument("--all-backbones", action="store_true",
                    help="include the over-budget backbones too")
    args = ap.parse_args()

    names = available_backbones() if args.all_backbones else args.backbones

    print(f"\nParameter budget: cap = {args.cap:,}\n")
    hdr = (f"{'backbone':<20}{'proj':>6}{'backbone':>11}{'proj':>9}"
           f"{'AMSM':>9}{'head':>9}{'TOTAL':>11}  fits")
    print(hdr)
    print("-" * len(hdr))

    rows: List[dict] = []
    for nm in names:
        for proj in args.proj:
            r = build_row(nm, proj, args.cap)
            if r is None:
                continue
            if "error" in r:
                print(f"{nm:<20}{proj:>6}  {r['error']}")
                continue
            rows.append(r)
            mark = "yes" if r["within_cap"] else "NO"
            print(f"{nm:<20}{proj:>6}{r['backbone']:>11,}{r['projection']:>9,}"
                  f"{r['amsm']:>9,}{r['head']:>9,}{r['total']:>11,}  {mark}")
        print()

    fitting = [r for r in rows if r["within_cap"]]
    if fitting:
        best = max(fitting, key=lambda r: r["total"])
        print("-" * len(hdr))
        print(f"Largest configuration that fits: {best['name']} "
              f"@ proj={best['proj']} -> {best['total']:,} parameters "
              f"({100 * best['total'] / args.cap:.0f}% of cap)")
        print(f"  stage channels {best['stage_in']} -> {best['stage_out']}, "
              f"descriptor {best['feat_dim']}-d")

        default = next((r for r in fitting
                        if r["name"] == MODEL.backbone
                        and r["proj"] == MODEL.proj_channels), None)
        if default:
            print(f"Current default ({MODEL.backbone} @ proj="
                  f"{MODEL.proj_channels}): {default['total']:,} parameters "
                  f"({100 * default['total'] / args.cap:.0f}% of cap)")

        print("\n  'Largest that fits' is not a recommendation -- PV has only "
              "380 original\n  training images, so spending the whole budget "
              "invites overfitting. Treat\n  the cap as a ceiling, not a "
              "target, and let the ablation decide.")

    # --- context ----------------------------------------------------------
    print("\n" + "=" * 68)
    print("PV benchmark for reference (Tables 7, 8, 11)")
    print("=" * 68)
    print(f"{'model':<22}{'params(M)':>11}{'PV F1':>9}{'pillQC F1':>11}{'gap':>9}")
    print("-" * 68)
    for nm, p, f_pv, f_pq in PV_BASELINES:
        flag = "  <- under cap" if p * 1e6 <= args.cap else ""
        print(f"{nm:<22}{p:>11.2f}{f_pv:>9.2f}{f_pq:>11.2f}"
              f"{f_pq - f_pv:>9.2f}{flag}")
    print("-" * 68)
    print("""
Reading this table before choosing a target:

  Only ShuffleNetV2 (1.26M) and MobileNetV3-small (1.52M) sit under 5M among
  PV's twelve, and both transfer badly -- they lose 35.34 and 26.45 points of
  F1 on pillQC. GoogLeNet at 5.60M loses only 6.87 and has the best
  cross-dataset F1 in the whole benchmark (76.19).

  So the honest target for a sub-5M model is NOT in-domain F1, where the
  leading group's intervals already overlap and nothing is statistically
  separable. It is cross-dataset F1 at a fraction of GoogLeNet's size, plus
  the uncertainty capability none of the twelve has. State the parameter count
  prominently -- it is the part of this work that is unambiguously better.""")


if __name__ == "__main__":
    main()
