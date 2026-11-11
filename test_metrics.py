"""
Validation of metrics.py against externally published values.

The PV paper reports point metrics AND their intervals for all twelve
benchmarked architectures, which is enough to reconstruct each confusion matrix
and then check our Wilson, bootstrap, McNemar and power-floor implementations
against numbers we did not produce. This is a stronger test than self-
consistency: if our Wilson interval disagrees with the paper's for the same
counts, one of the two is wrong.

    python test_metrics.py
"""
from __future__ import annotations

import numpy as np

from metrics import (
    auroc,
    bootstrap_f1_interval,
    compute_metrics,
    expected_calibration_error,
    fbeta,
    holm_correction,
    mcnemar_exact,
    metric_intervals,
    min_attainable_mcnemar_p,
    misclassification_detection,
    selective_risk_coverage,
    wilson_interval,
)

PASS, FAIL = [], []


def check(name: str, got, want, tol=1e-9):
    ok = bool(np.allclose(np.asarray(got, dtype=float),
                          np.asarray(want, dtype=float), atol=tol))
    (PASS if ok else FAIL).append(name)
    flag = "ok  " if ok else "FAIL"
    print(f"  [{flag}] {name}\n         got {got}\n         want {want}")


def scores_from_confusion(tp: int, tn: int, fp: int, fn: int, thr: float = 0.40):
    """Build labels + scores reproducing an exact confusion matrix.

    Scores are placed either side of the threshold, so compute_metrics has to
    apply the threshold correctly to recover the intended counts.
    """
    y, s = [], []
    y += [1] * tp; s += [thr + 0.3] * tp      # true positives
    y += [1] * fn; s += [thr - 0.3] * fn      # false negatives
    y += [0] * fp; s += [thr + 0.2] * fp      # false positives
    y += [0] * tn; s += [thr - 0.2] * tn      # true negatives
    return np.array(y), np.array(s)


# ---------------------------------------------------------------------------
print("\n=== 1. Point metrics vs PV Table 7 (EfficientNetV2-S) ===")
# Published: acc 88.50, prec 88.12, rec 89.00, F1 88.56 on n=200, 100 positive.
# Reconstructed confusion matrix: TP=89 FN=11 FP=12 TN=88.
y, s = scores_from_confusion(tp=89, tn=88, fp=12, fn=11)
m = compute_metrics(y, s, 0.40)
check("accuracy", m.accuracy * 100, 88.50, 0.01)
check("precision", m.precision * 100, 88.12, 0.01)
check("recall", m.recall * 100, 89.00, 0.01)
check("F1", m.f1 * 100, 88.56, 0.01)
check("confusion recovered", [m.tp, m.tn, m.fp, m.fn], [89, 88, 12, 11])

print("\n=== 2. Wilson intervals vs PV Table 9a (EfficientNetV2-S) ===")
ci = metric_intervals(m)
check("acc  CI", [v * 100 for v in ci["accuracy"]], [83.3, 92.2], 0.06)
check("prec CI", [v * 100 for v in ci["precision"]], [80.4, 93.1], 0.06)
check("rec  CI", [v * 100 for v in ci["recall"]], [81.4, 93.7], 0.06)

print("\n=== 3. Wilson intervals vs PV Table 9a (ViT-B/32) ===")
# Published: acc 82.00, prec 82.65, rec 81.00, F1 81.82 -> TP=81 FN=19 FP=17 TN=83
y2, s2 = scores_from_confusion(tp=81, tn=83, fp=17, fn=19)
m2 = compute_metrics(y2, s2, 0.40)
check("F1", m2.f1 * 100, 81.82, 0.01)
ci2 = metric_intervals(m2)
check("acc  CI", [v * 100 for v in ci2["accuracy"]], [76.1, 86.7], 0.06)
check("prec CI", [v * 100 for v in ci2["precision"]], [74.0, 88.9], 0.06)
check("rec  CI", [v * 100 for v in ci2["recall"]], [72.2, 87.5], 0.06)

print("\n=== 4. Bootstrap F1 interval vs PV Table 9a ===")
# Paper: EfficientNetV2-S F1 95% CI 83.6-92.9; ViT-B/32 75.5-87.3.
# Resampling is stochastic and our per-image score assignment is not the
# paper's, so agreement to ~1.5 pp is the realistic bar.
lo, hi = bootstrap_f1_interval(y, s, 0.40, n_resamples=10_000)
check("EffNetV2-S F1 CI", [lo * 100, hi * 100], [83.6, 92.9], 1.5)
lo2, hi2 = bootstrap_f1_interval(y2, s2, 0.40, n_resamples=10_000)
check("ViT-B/32   F1 CI", [lo2 * 100, hi2 * 100], [75.5, 87.3], 1.5)

print("\n=== 5. McNemar power floor vs PV Sect. 3.3.2 ===")
# "a four-image difference cannot reach significance ... (minimum attainable
# p = 0.125)" and on pillQC "differed by only two correctly classified images
# (minimum attainable McNemar p = 0.50)".
check("floor, 4 discordant", min_attainable_mcnemar_p(4), 0.125)
check("floor, 2 discordant", min_attainable_mcnemar_p(2), 0.50)
check("floor, 0 discordant", min_attainable_mcnemar_p(0), 1.0)

print("\n=== 6. Exact McNemar against the binomial ===")
# n01=10, n10=2 -> p = 2*P(X<=2), X~Bin(12, 0.5) = 2*(1+12+66)/4096
a = np.array([True] * 10 + [False] * 2 + [True] * 50)
b = np.array([False] * 10 + [True] * 2 + [True] * 50)
r = mcnemar_exact(a, b)
check("n01/n10", [r["n01"], r["n10"]], [10, 2])
check("p-value", r["p_value"], 2 * (1 + 12 + 66) / 4096, 1e-12)
check("identical models -> p=1", mcnemar_exact(a, a)["p_value"], 1.0)

print("\n=== 7. Holm step-down ===")
# Worked by hand: sorted 0.01, 0.03, 0.04 -> raw 3*0.01, 2*0.03, 1*0.04
# = 0.03, 0.06, 0.04; the running maximum enforces monotonicity, so the last
# becomes 0.06, not 0.04. (A naive per-rank scaling without that step-down
# would give 0.08 for p=0.04 -- that is the usual error, and not Holm.)
check("holm([.01,.04,.03])", holm_correction([0.01, 0.04, 0.03]),
      [0.03, 0.06, 0.06], 1e-12)
check("holm preserves order", holm_correction([0.5, 0.001]), [0.5, 0.002], 1e-12)
check("holm clamps at 1", holm_correction([0.9, 0.8]), [1.0, 1.0], 1e-12)
# PV Table 10, PV-test-set rows: EfficientNetV2-S vs Xception / Swin-B /
# ShuffleNetV2 / AlexNet. Paper reports p_Holm = 1.00, 1.00, 1.00, <0.001.
check("PV Table 10 reproduced",
      holm_correction([0.57, 0.57, 0.47, 0.0001]), [1.0, 1.0, 1.0, 0.0004], 1e-12)

try:
    from statsmodels.stats.multitest import multipletests

    rng_h = np.random.default_rng(3)
    for _ in range(200):
        p = rng_h.random(rng_h.integers(2, 9)).round(4)
        if not np.allclose(multipletests(p, method="holm")[1], holm_correction(p)):
            FAIL.append("holm vs statsmodels (random)")
            print(f"  [FAIL] holm vs statsmodels on {p}")
            break
    else:
        PASS.append("holm vs statsmodels (200 random cases)")
        print("  [ok  ] holm vs statsmodels, 200 random cases")
except ImportError:
    print("  [skip] statsmodels not installed; cross-check skipped")

print("\n=== 8. AUROC ===")
check("perfect separation", auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
check("perfectly inverted", auroc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]), 0.0)
check("all ties -> 0.5", auroc([0, 0, 1, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)
# One tied pair across classes: 3 of 4 pairs ordered, 1 tied -> (3 + 0.5)/4
check("one tied pair", auroc([0, 0, 1, 1], [0.1, 0.5, 0.5, 0.9]), 0.875)
# Rank-based AUROC must equal the Mann-Whitney U statistic.
rng = np.random.default_rng(0)
lab = rng.integers(0, 2, 300)
sc = rng.normal(lab * 0.8, 1.0)
pos, neg = sc[lab == 1], sc[lab == 0]
mw = np.mean((pos[:, None] > neg[None, :]) + 0.5 * (pos[:, None] == neg[None, :]))
check("matches Mann-Whitney U", auroc(lab, sc), mw, 1e-12)

print("\n=== 9. F-beta weighting ===")
check("F1 == F-beta(1)", fbeta(0.8, 0.6, 1.0), 2 * 0.8 * 0.6 / 1.4, 1e-12)
check("F2 weights recall", fbeta(0.8, 0.6, 2.0), 5 * 0.8 * 0.6 / (4 * 0.8 + 0.6), 1e-12)
# The protocol's reason for using F2: at equal F1, higher recall must win.
hi_rec = fbeta(0.70, 0.90, 2.0)
hi_prec = fbeta(0.90, 0.70, 2.0)
check("F2 prefers recall over precision", hi_rec > hi_prec, True)
check("F1 is symmetric in the same pair",
      fbeta(0.70, 0.90, 1.0), fbeta(0.90, 0.70, 1.0), 1e-12)
check("zero precision -> 0", fbeta(0.0, 0.9, 2.0), 0.0)

print("\n=== 10. Threshold is actually applied ===")
yt = np.array([0, 0, 1, 1])
sc2 = np.array([0.35, 0.45, 0.45, 0.80])
check("thr 0.40 -> 1 FP, 0 FN",
      [compute_metrics(yt, sc2, 0.40).fp, compute_metrics(yt, sc2, 0.40).fn], [1, 0])
check("thr 0.50 -> 0 FP, 1 FN",
      [compute_metrics(yt, sc2, 0.50).fp, compute_metrics(yt, sc2, 0.50).fn], [0, 1])

print("\n=== 11. Uncertainty / selective prediction ===")
# Errors deliberately given the highest uncertainty -> AUROC must be 1.0.
y3 = np.array([0, 0, 1, 1, 1, 0])
s3 = np.array([0.1, 0.9, 0.9, 0.8, 0.1, 0.2])   # index 1 and 4 are errors
u3 = np.array([0.1, 0.9, 0.1, 0.2, 0.8, 0.1])
md = misclassification_detection(y3, s3, u3, 0.40)
check("n_errors", md["n_errors"], 2)
check("error-detection AUROC", md["auroc"], 1.0)
check("separation ratio > 1", md["separation_ratio"] > 1.0, True)

rc = selective_risk_coverage(y3, s3, u3, coverages=(1.0, 0.5), threshold=0.40)
check("full coverage error rate", rc[0]["error_rate"], 2 / 6, 1e-12)
check("50% coverage drops error to 0", rc[1]["error_rate"], 0.0)
check("coverage monotone: risk cannot rise when deferring",
      rc[1]["error_rate"] <= rc[0]["error_rate"], True)

print("\n=== 12. Calibration ===")
# Perfectly calibrated and perfectly confident -> ECE 0.
probs_perfect = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])
check("ECE of perfect confident predictions",
      expected_calibration_error([0, 1, 0], probs_perfect), 0.0, 1e-12)
# Fully confident and always wrong -> ECE 1.
check("ECE of confident-and-wrong",
      expected_calibration_error([1, 0, 1], probs_perfect), 1.0, 1e-12)

# ---------------------------------------------------------------------------
print("\n" + "=" * 62)
print(f"  {len(PASS)} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print(f"    FAILED: {f}")
    raise SystemExit(1)
print("  metrics.py agrees with the PV paper's published intervals,")
print("  power floors, and with independent analytic references.")
print("=" * 62)
