"""
Metrics and statistics, matching the PV paper's analysis (Sect. 2.4, 3.3).

  - accuracy / precision / recall / F1 / F2, defective = positive class
  - Wilson score intervals for the three proportions
  - percentile bootstrap intervals for F1 (10,000 resamples of test images)
  - exact McNemar test on per-image correctness
  - paired bootstrap interval for an F1 difference (same resampled images)
  - Holm step-down correction across a family of comparisons
  - ECE, Brier, misclassification-detection AUROC, selective risk-coverage

All thresholded metrics take an explicit threshold. The PV protocol fixes it
at 0.40 for both the in-domain test set and pillQC; do not retune it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import PV


# ---------------------------------------------------------------------------
# Point metrics
# ---------------------------------------------------------------------------
@dataclass
class ClassificationMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float
    f2: float
    tp: int
    tn: int
    fp: int
    fn: int

    def as_dict(self) -> dict:
        return asdict(self)

    def pretty(self, pct: bool = True) -> str:
        s = 100.0 if pct else 1.0
        return (f"acc {self.accuracy * s:.2f}  prec {self.precision * s:.2f}  "
                f"rec {self.recall * s:.2f}  F1 {self.f1 * s:.2f}  F2 {self.f2 * s:.2f}")


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return tp, tn, fp, fn


def fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return 0.0 if denom == 0 else (1 + b2) * precision * recall / denom


def compute_metrics(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float = PV.decision_threshold,
) -> ClassificationMetrics:
    """`scores` are P(defective). Positive class is 'defective' (index 1)."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    tp, tn, fp, fn = confusion(y_true, y_pred)

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    return ClassificationMetrics(
        accuracy=acc, precision=prec, recall=rec,
        f1=fbeta(prec, rec, 1.0), f2=fbeta(prec, rec, 2.0),
        tp=tp, tn=tn, fp=fp, fn=fn,
    )


# ---------------------------------------------------------------------------
# Interval estimation
# ---------------------------------------------------------------------------
def wilson_interval(k: int, n: int, conf: float = PV.confidence_level) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion (Wilson 1927)."""
    if n == 0:
        return (float("nan"), float("nan"))
    from statistics import NormalDist

    z = NormalDist().inv_cdf(0.5 + conf / 2.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def metric_intervals(
    m: ClassificationMetrics, conf: float = PV.confidence_level
) -> Dict[str, Tuple[float, float]]:
    n = m.tp + m.tn + m.fp + m.fn
    return {
        "accuracy": wilson_interval(m.tp + m.tn, n, conf),
        "precision": wilson_interval(m.tp, m.tp + m.fp, conf),
        "recall": wilson_interval(m.tp, m.tp + m.fn, conf),
    }


def bootstrap_f1_interval(
    y_true: Sequence[int],
    scores: Sequence[float],
    threshold: float = PV.decision_threshold,
    n_resamples: int = PV.bootstrap_resamples,
    conf: float = PV.confidence_level,
    seed: int = PV.seed,
) -> Tuple[float, float]:
    """Percentile bootstrap over test IMAGES. F1 has no closed-form variance."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    vals = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        m = compute_metrics(y_true[idx], scores[idx], threshold)
        vals[i] = m.f1
    lo = (1 - conf) / 2 * 100
    return (float(np.percentile(vals, lo)), float(np.percentile(vals, 100 - lo)))


# ---------------------------------------------------------------------------
# Paired comparisons
# ---------------------------------------------------------------------------
def mcnemar_exact(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    """Exact (binomial) McNemar test on per-image correctness.

    Returns n01 (A only correct), n10 (B only correct) and the two-sided p.
    The exact test is used rather than the chi-square approximation because the
    PV test set is small and discordant counts can be in single digits.
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p_value": 1.0}

    from math import comb

    k = min(n01, n10)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return {"n01": n01, "n10": n10, "p_value": float(min(1.0, 2 * tail))}


def min_attainable_mcnemar_p(n_discordant: int) -> float:
    """Smallest p the exact test can return for a given discordant count.

    Useful for stating honestly that two models differing by a handful of
    images CANNOT reach significance however the errors fall. The PV paper
    makes exactly this point (Sect. 3.3.2).
    """
    if n_discordant == 0:
        return 1.0
    return float(min(1.0, 2 * (0.5 ** n_discordant)))


def paired_bootstrap_f1_diff(
    y_true: Sequence[int],
    scores_a: Sequence[float],
    scores_b: Sequence[float],
    threshold: float = PV.decision_threshold,
    n_resamples: int = PV.bootstrap_resamples,
    conf: float = PV.confidence_level,
    seed: int = PV.seed,
) -> Dict[str, float]:
    """CI for F1(A) - F1(B), drawing the SAME resampled images for both models."""
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    n = len(y_true)
    rng = np.random.default_rng(seed)

    diffs = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        yt = y_true[idx]
        diffs[i] = compute_metrics(yt, a[idx], threshold).f1 - compute_metrics(yt, b[idx], threshold).f1
    lo = (1 - conf) / 2 * 100
    point = compute_metrics(y_true, a, threshold).f1 - compute_metrics(y_true, b, threshold).f1
    return {
        "delta_f1": float(point),
        "ci_low": float(np.percentile(diffs, lo)),
        "ci_high": float(np.percentile(diffs, 100 - lo)),
    }


def holm_correction(p_values: Sequence[float]) -> List[float]:
    """Holm step-down adjustment, preserving input order (Holm 1979)."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]
        running = max(running, val)
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


# ---------------------------------------------------------------------------
# Calibration and uncertainty
# ---------------------------------------------------------------------------
def expected_calibration_error(
    y_true: Sequence[int], probs: np.ndarray, n_bins: int = 15
) -> float:
    """ECE over equal-width confidence bins, weighted by bin occupancy.

    Caveat worth carrying into any write-up: standard ECE bins by MAXIMUM
    softmax confidence. For a Dirichlet-parameterized head, p = alpha / S is
    expected Dirichlet mass rather than a confidence score, so ECE tends to
    read higher for EDL than for softmax models even at equal discrimination.
    Report it, but judge uncertainty quality by misclassification AUROC and
    the separation ratio.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def brier_score(y_true: Sequence[int], probs: np.ndarray) -> float:
    """Multiclass Brier: per-sample squared error against the one-hot outcome."""
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs, dtype=float)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(((probs - onehot) ** 2).sum(axis=1).mean())


def auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Rank-based AUROC with tie handling. No sklearn dependency."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(scores, kind="mergesort")
    s_sorted = scores[order]
    ranks = np.empty(len(scores), dtype=float)
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def misclassification_detection(
    y_true: Sequence[int],
    scores: Sequence[float],
    uncertainty: Sequence[float],
    threshold: float = PV.decision_threshold,
) -> Dict[str, float]:
    """Can the uncertainty signal rank the model's own errors?

    Prediction ERROR is the positive class, so AUROC > 0.5 means higher
    uncertainty on mistakes. Also reports the separation ratio (mean
    uncertainty on errors / mean on correct predictions).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    u = np.asarray(uncertainty, dtype=float)
    err = (y_pred != y_true).astype(int)
    if err.sum() == 0:
        return {"auroc": float("nan"), "separation_ratio": float("nan"), "n_errors": 0}

    mu_err = float(u[err == 1].mean())
    mu_ok = float(u[err == 0].mean()) if (err == 0).any() else float("nan")
    return {
        "auroc": auroc(err, u),
        "separation_ratio": mu_err / mu_ok if mu_ok else float("nan"),
        "mean_uncertainty_error": mu_err,
        "mean_uncertainty_correct": mu_ok,
        "n_errors": int(err.sum()),
    }


def selective_risk_coverage(
    y_true: Sequence[int],
    scores: Sequence[float],
    uncertainty: Sequence[float],
    coverages: Sequence[float] = (1.0, 0.95, 0.9, 0.8, 0.5),
    threshold: float = PV.decision_threshold,
) -> List[Dict[str, float]]:
    """Error rate when the most uncertain samples are deferred to a human.

    This is the operationally relevant view for tablet inspection: a model that
    abstains on its least certain cases and routes them to a QA inspector.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    u = np.asarray(uncertainty, dtype=float)
    order = np.argsort(u)          # most certain first
    n = len(y_true)

    rows = []
    for c in coverages:
        k = max(1, int(round(c * n)))
        keep = order[:k]
        err = float((y_pred[keep] != y_true[keep]).mean())
        rec_den = int((y_true[keep] == 1).sum())
        rows.append({
            "coverage": float(k / n),
            "n_retained": k,
            "error_rate": err,
            "selective_accuracy": 1.0 - err,
            "recall_on_retained": (
                float(((y_pred[keep] == 1) & (y_true[keep] == 1)).sum() / rec_den)
                if rec_den else float("nan")
            ),
        })
    return rows
