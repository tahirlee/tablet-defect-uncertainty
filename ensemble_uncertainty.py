r"""
Uncertainty for the proposed model: a 3-seed deep ensemble of ConvNeXt-Tiny AMSM + CE.

CPU only. It reads the predictions saved by run_all.py, evaluate.py and
save_val_predictions.py, so nothing is retrained. Run from the experiments folder
after save_val_predictions.py has been run:

    python ensemble_uncertainty.py

Score chosen before looking at results: total predictive entropy of the averaged
defective probability. Seed disagreement (epistemic part) is reported as a secondary score.

Deferral rate is chosen on the VALIDATION set only, by a rule fixed in advance:
among deferral rates 0%, 10% and 20%, pick the one with the highest accuracy on the
tablets kept (ties go to the smaller rate). That single rate is then applied once to
PV test and pillQC. The test-set curves are printed for reference only.

Error detection is measured with AUROC for ranking the model's mistakes. The ensemble
score is compared with the mean single-seed score, 1 - max probability. A paired
bootstrap over images gives a 95% interval for the difference. Results are written to
ensemble_uncertainty.txt.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from config import PATHS, PV, run_dir_name
from metrics import auroc, compute_metrics

SEEDS = (7, 42, 123)
RESULTS = "ensemble_uncertainty.txt"
EPS = 1e-12
DEFER_CAP_GRID = (0.0, 0.1, 0.2)        # validation chooses from these only
DEFER_TABLE = (0.0, 0.1, 0.2, 0.3)      # printed for reference
BOOT_N = 10_000
BOOT_SEED = 0


def run_dir(seed: int) -> Path:
    return Path(PATHS.output_dir) / run_dir_name("convnext_tiny", "amsm_only", "ce", seed)


def entropy(p: np.ndarray) -> np.ndarray:
    """Binary entropy in nats."""
    p = np.clip(p, EPS, 1.0 - EPS)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def load_split(file_name: str):
    labels, scores = None, []
    for seed in SEEDS:
        f = run_dir(seed) / file_name
        if not f.exists():
            raise SystemExit(
                f"Missing {f}. Run run_all.py first, and save_val_predictions.py for val_predictions.npz."
            )
        d = np.load(f, allow_pickle=True)
        lab = d["labels"].astype(int)
        if labels is None:
            labels = lab
        elif not np.array_equal(labels, lab):
            raise SystemExit(f"Labels differ between seeds in {file_name}; the splits do not match.")
        scores.append(d["scores"].astype(float))
    return labels, np.stack(scores)          # P has shape [seeds, images]


def ensemble_errors(labels: np.ndarray, P: np.ndarray):
    thr = PV.decision_threshold
    pbar = P.mean(axis=0)
    err = ((pbar >= thr).astype(int) != labels).astype(int)
    return pbar, err


def choose_deferral_on_validation(labels: np.ndarray, P: np.ndarray) -> float:
    _, err = ensemble_errors(labels, P)
    H = entropy(P.mean(axis=0))
    order = np.argsort(H, kind="stable")
    best_f, best_acc = None, -1.0
    for f in DEFER_CAP_GRID:
        k = int(round((1.0 - f) * len(labels)))
        acc = 1.0 - err[order[:k]].mean()
        if acc > best_acc + 1e-12:              # strict: ties keep the smaller rate
            best_f, best_acc = f, acc
    return best_f


def bootstrap_auroc_gain(err, H, single_scores):
    """Paired bootstrap over images: ensemble AUROC minus mean single-seed AUROC."""
    rng = np.random.default_rng(BOOT_SEED)
    n = len(err)
    diffs = []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        e = err[idx]
        if e.min() == e.max():                  # AUROC undefined without both classes
            continue
        a_ens = auroc(e, H[idx])
        a_single = float(np.mean([auroc(e, s[idx]) for s in single_scores]))
        diffs.append(a_ens - a_single)
    d = np.array(diffs)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), float((d > 0).mean()), len(d)


def analyse(title: str, labels: np.ndarray, P: np.ndarray, defer_frac: float,
            out: list, bootstrap: bool) -> None:
    thr = PV.decision_threshold
    pbar, err_ens = ensemble_errors(labels, P)

    H_total = entropy(pbar)
    H_epistemic = H_total - entropy(P).mean(axis=0)   # mutual information across seeds

    out.append(f"== {title}  (n = {len(labels)}, defective {100 * labels.mean():.1f}%) ==")

    single_scores, single_aurocs = [], []
    for i, seed in enumerate(SEEDS):
        m = compute_metrics(labels, P[i], thr)
        err = ((P[i] >= thr).astype(int) != labels).astype(int)
        score = np.minimum(P[i], 1.0 - P[i])            # 1 - max probability, binary
        single_scores.append(score)
        a = auroc(err, score)
        single_aurocs.append(a)
        out.append(f"  seed {seed:<4}  acc {100 * m.accuracy:6.2f}  F1 {100 * m.f1:6.2f}  "
                   f"errors {err.sum():3d}  error AUROC (1 - max p) {a:.4f}")

    m = compute_metrics(labels, pbar, thr)
    out.append(f"  ensemble   acc {100 * m.accuracy:6.2f}  F1 {100 * m.f1:6.2f}  "
               f"errors {err_ens.sum():3d}")

    a_total = auroc(err_ens, H_total)
    a_epi = auroc(err_ens, H_epistemic)
    mean_single = float(np.mean(single_aurocs))
    out.append(f"  error AUROC, total entropy (pre-declared)     {a_total:.4f}")
    out.append(f"  error AUROC, seed disagreement (epistemic)   {a_epi:.4f}")
    out.append(f"  error AUROC, mean single seed (1 - max p)    {mean_single:.4f}")
    out.append(f"  total entropy beats the mean single-seed score: "
               f"{'YES' if a_total > mean_single else 'NO'}")

    if bootstrap:
        mean_d, lo, hi, p_pos, nb = bootstrap_auroc_gain(err_ens, H_total, single_scores)
        out.append(f"  paired bootstrap, AUROC gain (ensemble - mean single): "
                   f"{mean_d:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
                   f"P(gain > 0) = {p_pos:.3f}  ({nb} valid resamples)")

    order = np.argsort(H_total, kind="stable")          # least uncertain first
    k = int(round((1.0 - defer_frac) * len(labels)))
    kept = order[:k]
    out.append(f"  chosen on validation: defer {int(round(100 * defer_frac))}%  "
               f"kept {k}  accuracy on kept {100 * (1.0 - err_ens[kept].mean()):6.2f}")

    out.append("  deferral table (reference only):")
    for f in DEFER_TABLE:
        kk = int(round((1.0 - f) * len(labels)))
        acc_keep = 100 * (1.0 - err_ens[order[:kk]].mean())
        out.append(f"    defer {int(round(100 * f)):>2}%  kept {kk:3d}  accuracy on kept {acc_keep:6.2f}")
    out.append("")


def main() -> None:
    out = [
        "Uncertainty for ConvNeXt-Tiny AMSM + CE, 3-seed deep ensemble",
        "Score fixed before results: total predictive entropy of the averaged probability.",
        "Deferral rate chosen on validation only (grid 0/10/20%, ties to the smaller rate).",
        "",
    ]

    v_labels, V = load_split("val_predictions.npz")
    defer = choose_deferral_on_validation(v_labels, V)
    pbar_v, err_v = ensemble_errors(v_labels, V)
    out.append(f"== Validation (n = {len(v_labels)}) : used only to choose the deferral rate ==")
    out.append(f"  ensemble acc {100 * (1 - err_v.mean()):6.2f}  errors {err_v.sum()}")
    out.append(f"  chosen deferral rate: {int(round(100 * defer))}%")
    out.append("")

    labels, P = load_split("test_predictions.npz")
    analyse("PV test set", labels, P, defer, out, bootstrap=True)

    labels, P = load_split("pillqc_predictions.npz")
    analyse("pillQC zero-adaptation transfer", labels, P, defer, out, bootstrap=True)

    text = "\n".join(out)
    print(text)
    Path(RESULTS).write_text(text + "\n", encoding="utf-8")
    print(f"\n[saved] {RESULTS}")


if __name__ == "__main__":
    main()
