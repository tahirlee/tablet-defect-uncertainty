"""
Evaluation: in-domain PV test set, zero-adaptation transfer to pillQC,
statistics, and uncertainty quality.

Usage
    python evaluate.py --ckpt runs/convnext_tiny_amsm_only_edl_seed7/best.pt \
                       --train-root data/PV_augmented --pillqc data/pillQC

    # paired comparison against a second run (e.g. the CE variant, or a baseline)
    python evaluate.py --ckpt runs/convnext_tiny_amsm_only_edl_seed7/best.pt \
                       --compare runs/convnext_tiny_amsm_only_ce_seed7/test_predictions.npz \
                       --train-root data/PV_augmented
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from config import MODEL, PV, PATHS
from data import make_loaders, make_pillqc_loader, seed_everything
from metrics import (
    bootstrap_f1_interval,
    brier_score,
    compute_metrics,
    expected_calibration_error,
    holm_correction,
    mcnemar_exact,
    metric_intervals,
    min_attainable_mcnemar_p,
    misclassification_detection,
    paired_bootstrap_f1_diff,
    selective_risk_coverage,
)
from model import build_model
from train import run_inference


# ---------------------------------------------------------------------------
def load_checkpoint(path: str | Path, device) -> torch.nn.Module:
    """Rebuild the exact architecture the checkpoint was trained with.

    Fallbacks come from config.MODEL, never from literals, so an old checkpoint
    without a stored key is rebuilt with the current defaults rather than a
    stale backbone name.
    """
    ck = torch.load(path, map_location=device)
    model = build_model(
        variant=ck.get("variant", "amsm_only"),
        head_type=ck.get("head", MODEL.head_type),
        backbone=ck.get("backbone", MODEL.backbone),
        strides=tuple(ck.get("strides", MODEL.feature_strides)),
        proj_channels=ck.get("proj_channels", MODEL.proj_channels),
        amsm_norm=ck.get("amsm_norm"),
        num_classes=2,
        pretrained=False,   # weights come from the checkpoint
    )
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


@torch.no_grad()
def mc_dropout_inference(model, loader, device, passes: int = 20) -> Dict[str, np.ndarray]:
    """MC-Dropout uncertainty baseline.

    Only Dropout layers are put into train mode, so the MAP prediction is
    unchanged; the stochastic passes supply predictive entropy and mutual
    information. Costs `passes` forward passes per image -- the comparison the
    single-pass evidential head is meant to beat on latency, not necessarily on
    AUROC.
    """
    model.enable_mc_dropout()
    all_probs, labels, paths = [], [], []

    for x, y, p in loader:
        x = x.to(device, non_blocking=True)
        stack = torch.stack([model(x)["probs"] for _ in range(passes)])  # [T,B,K]
        all_probs.append(stack.cpu().numpy())
        labels.append(np.asarray(y))
        paths.extend(p)

    probs_t = np.concatenate(all_probs, axis=1)        # [T, N, K]
    mean = probs_t.mean(axis=0)
    eps = 1e-12
    pred_entropy = -(mean * np.log(mean + eps)).sum(axis=1)
    exp_entropy = -(probs_t * np.log(probs_t + eps)).sum(axis=2).mean(axis=0)

    model.eval()
    return {
        "probs": mean,
        "scores": mean[:, 1],
        "predictive_entropy": pred_entropy,
        "mutual_information": pred_entropy - exp_entropy,
        "labels": np.concatenate(labels),
        "paths": np.array(paths),
    }


# ---------------------------------------------------------------------------
def evaluate_split(pred: Dict[str, np.ndarray], name: str,
                   threshold: float = PV.decision_threshold) -> dict:
    y, s, u = pred["labels"], pred["scores"], pred["uncertainty"]
    m = compute_metrics(y, s, threshold)
    ci = metric_intervals(m)
    f1_ci = bootstrap_f1_interval(y, s, threshold)

    report = {
        "split": name,
        "n": int(len(y)),
        "n_positive": int(y.sum()),
        "prevalence": float(y.mean()),
        "threshold": threshold,
        "metrics": m.as_dict(),
        "wilson_ci": {k: list(v) for k, v in ci.items()},
        "f1_bootstrap_ci": list(f1_ci),
        "ece": expected_calibration_error(y, pred["probs"]),
        "brier": brier_score(y, pred["probs"]),
        "uncertainty": misclassification_detection(y, s, u, threshold),
        "selective_risk": selective_risk_coverage(y, s, u, threshold=threshold),
    }

    pct = lambda v: f"{v * 100:.2f}"
    print(f"\n--- {name} (n={len(y)}, {y.mean() * 100:.1f}% defective) ---")
    print(f"  accuracy  {pct(m.accuracy)}  [{pct(ci['accuracy'][0])}-{pct(ci['accuracy'][1])}]")
    print(f"  precision {pct(m.precision)}  [{pct(ci['precision'][0])}-{pct(ci['precision'][1])}]")
    print(f"  recall    {pct(m.recall)}  [{pct(ci['recall'][0])}-{pct(ci['recall'][1])}]")
    print(f"  F1        {pct(m.f1)}  [{pct(f1_ci[0])}-{pct(f1_ci[1])}]")
    print(f"  confusion tp={m.tp} tn={m.tn} fp={m.fp} fn={m.fn}")
    print(f"  ECE {report['ece']:.4f}   Brier {report['brier']:.4f}")
    un = report["uncertainty"]
    if un["n_errors"]:
        print(f"  misclassification AUROC {un['auroc']:.4f}  "
              f"separation {un['separation_ratio']:.2f}x  ({un['n_errors']} errors)")
    return report


def compare_runs(y_true, scores_a, scores_b, name_a: str, name_b: str,
                 threshold: float = PV.decision_threshold) -> dict:
    pred_a = (np.asarray(scores_a) >= threshold).astype(int)
    pred_b = (np.asarray(scores_b) >= threshold).astype(int)
    y = np.asarray(y_true).astype(int)

    mc = mcnemar_exact(pred_a == y, pred_b == y)
    boot = paired_bootstrap_f1_diff(y, scores_a, scores_b, threshold)
    n_disc = mc["n01"] + mc["n10"]
    floor = min_attainable_mcnemar_p(n_disc)

    out = {
        "model_a": name_a, "model_b": name_b,
        "correct_a": int((pred_a == y).sum()), "correct_b": int((pred_b == y).sum()),
        **mc, **boot,
        "min_attainable_p": floor,
        "underpowered": floor > 0.05,
    }
    print(f"\n{name_a} vs {name_b}: "
          f"{out['correct_a']}/{len(y)} vs {out['correct_b']}/{len(y)} correct; "
          f"dF1 {boot['delta_f1'] * 100:+.2f} pp "
          f"[{boot['ci_low'] * 100:+.2f}, {boot['ci_high'] * 100:+.2f}]; "
          f"McNemar p={mc['p_value']:.4f}")
    if out["underpowered"]:
        print(f"  note: only {n_disc} discordant images -- the exact test cannot "
              f"return p < {floor:.3f} however the errors fall. Non-significance "
              f"here reflects test-set size, not architectural equivalence.")
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-root", default=PATHS.pv_augmented_root)
    ap.add_argument("--pillqc", default=PATHS.pillqc_root)
    ap.add_argument("--skip-pillqc", action="store_true")
    ap.add_argument("--mc-dropout", type=int, default=0,
                    help="number of MC-Dropout passes for the uncertainty baseline (0 = skip)")
    ap.add_argument("--compare", nargs="*", default=[],
                    help="test_predictions.npz files to compare against, paired")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    seed_everything(PV.seed)
    device = torch.device(args.device if args.device else
                          ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_checkpoint(args.ckpt, device)
    report: dict = {"checkpoint": str(args.ckpt), "threshold": PV.decision_threshold}

    # -- in-domain -----------------------------------------------------------
    _, _, test_loader = make_loaders(args.train_root, PV.batch_size, args.num_workers)
    pv_pred = run_inference(model, test_loader, device)
    report["pv_test"] = evaluate_split(pv_pred, "PV test set")

    # -- cross-dataset, zero adaptation --------------------------------------
    if not args.skip_pillqc:
        try:
            pq_loader = make_pillqc_loader(args.pillqc, PV.batch_size, args.num_workers)
            pq_pred = run_inference(model, pq_loader, device)
            report["pillqc"] = evaluate_split(pq_pred, "pillQC (zero adaptation)")

            d = (report["pv_test"]["metrics"]["f1"] - report["pillqc"]["metrics"]["f1"])
            dr = (report["pv_test"]["metrics"]["recall"] - report["pillqc"]["metrics"]["recall"])
            report["generalization_gap"] = {"f1_pp": d * 100, "recall_pp": dr * 100}
            print(f"\n[transfer] F1 drops {d * 100:.2f} pp, recall drops {dr * 100:.2f} pp")
            print("  Note: pillQC prevalence is 22.4% vs 50% on the PV test set, so part "
                  "of the F1 and precision change is a base-rate effect. Recall is "
                  "prevalence-independent and is the figure to read for a screening claim.")
            np.savez_compressed(Path(args.ckpt).parent / "pillqc_predictions.npz", **pq_pred)
        except FileNotFoundError as exc:
            print(f"[skip] pillQC: {exc}")

    # -- MC-Dropout uncertainty baseline -------------------------------------
    if args.mc_dropout > 0:
        mc = mc_dropout_inference(model, test_loader, device, args.mc_dropout)
        report["mc_dropout"] = {
            "passes": args.mc_dropout,
            "predictive_entropy": misclassification_detection(
                mc["labels"], mc["scores"], mc["predictive_entropy"]),
            "mutual_information": misclassification_detection(
                mc["labels"], mc["scores"], mc["mutual_information"]),
            "single_pass_auroc": report["pv_test"]["uncertainty"]["auroc"],
        }
        print(f"\n[MC-Dropout, T={args.mc_dropout}] "
              f"pred-entropy AUROC "
              f"{report['mc_dropout']['predictive_entropy']['auroc']:.4f} vs "
              f"single-pass evidential "
              f"{report['pv_test']['uncertainty']['auroc']:.4f}")

    # -- paired comparisons --------------------------------------------------
    if args.compare:
        comparisons, p_values = [], []
        for other in args.compare:
            d = np.load(other, allow_pickle=True)
            if not np.array_equal(d["labels"], pv_pred["labels"]):
                print(f"[skip] {other}: label order differs; cannot pair")
                continue
            c = compare_runs(pv_pred["labels"], pv_pred["scores"], d["scores"],
                             Path(args.ckpt).parent.name, Path(other).parent.name)
            comparisons.append(c)
            p_values.append(c["p_value"])

        if comparisons:
            for c, adj in zip(comparisons, holm_correction(p_values)):
                c["p_holm"] = adj
                c["significant"] = adj < 0.05
            report["comparisons"] = comparisons
            print("\n[Holm-adjusted across "
                  f"{len(comparisons)} comparisons]")
            for c in comparisons:
                flag = "significant" if c["significant"] else "NS"
                print(f"  vs {c['model_b']}: p={c['p_value']:.4f} -> "
                      f"p_holm={c['p_holm']:.4f} ({flag})")

    out_path = Path(args.out) if args.out else Path(args.ckpt).parent / "evaluation.json"
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
