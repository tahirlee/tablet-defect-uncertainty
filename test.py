"""
Test UAMSF-Net and the three-network ensemble.

1. Test one trained network on PV and pillQC (saves its predictions):

    python test.py --ckpt runs/seed7/best.pt --data data/PV_augmented --pillqc data/pillQC/images

   Run it for seeds 7, 42 and 123. It writes val_predictions.npz,
   test_predictions.npz and pillqc_predictions.npz next to the checkpoint.

2. Ensemble analysis from the saved predictions (CPU, no GPU needed):

    python test.py --ensemble runs/seed7 runs/seed42 runs/seed123

   Reports accuracy, precision, recall and F1 (Wilson 95% intervals for the
   ensemble), error-detection AUROC of the predictive entropy and of the
   mutual information, the two comparisons with the members (same-target and
   own-error, paired bootstrap, 10,000 resamples), selective review with the
   deferral rate chosen on validation, and ensembles of 1, 2 and 3 members.
   Predictions of different networks are matched by image file before they
   are combined. Results are written to ensemble_results.txt.

3. Latency of one network and of the ensemble (GPU, batch size 1):

    python test.py --latency runs/seed7/best.pt runs/seed42/best.pt runs/seed123/best.pt
"""
from __future__ import annotations

import argparse
import itertools
import math
import re
import time
from pathlib import Path
from statistics import NormalDist

import numpy as np
import torch

from data import list_pillqc, list_pv, make_loader
from model import load_model
from train import THRESHOLD, predict

EPS = 1e-12
DEFER_GRID = (0.0, 0.1, 0.2)          # deferral rates the validation set chooses from
DEFER_TABLE = (0.0, 0.1, 0.2, 0.3)    # rates reported for reference
BOOT_N, BOOT_SEED = 10_000, 0


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def confusion(labels, pred):
    return {"TP": int(((pred == 1) & (labels == 1)).sum()), "TN": int(((pred == 0) & (labels == 0)).sum()),
            "FP": int(((pred == 1) & (labels == 0)).sum()), "FN": int(((pred == 0) & (labels == 1)).sum())}


def metrics(labels, probs):
    c = confusion(labels, (probs >= THRESHOLD).astype(int))
    tp, tn, fp, fn = c["TP"], c["TN"], c["FP"], c["FN"]
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {**c, "accuracy": (tp + tn) / len(labels), "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}


def wilson(k: int, n: int, conf: float = 0.95):
    if n == 0:
        return float("nan"), float("nan")
    z = NormalDist().inv_cdf(0.5 + conf / 2)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def auroc(labels, scores) -> float:
    """Rank-based AUROC with ties; labels = 1 for the positive event."""
    labels, scores = np.asarray(labels).astype(int), np.asarray(scores, dtype=float)
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    s = scores[order]
    ranks = np.empty(len(scores))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def entropy(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def errors(labels, probs):
    return ((probs >= THRESHOLD).astype(int) != labels).astype(int)


# ---------------------------------------------------------------------------
# Mode 1: test one network
# ---------------------------------------------------------------------------
def test_one(a) -> None:
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = load_model(a.ckpt, device)
    pv = list_pv(a.data)
    out_dir = Path(a.ckpt).parent
    sets = [("val", pv["val"]), ("test", pv["test"])]
    if a.pillqc:
        sets.append(("pillqc", list_pillqc(a.pillqc)))
    for name, items in sets:
        pred = predict(model, make_loader(items, 16, num_workers=a.workers), device)
        np.savez_compressed(out_dir / f"{name}_predictions.npz", **pred)
        m = metrics(pred["labels"], pred["scores"])
        print(f"{name:<7} n={len(items):<4} accuracy {100 * m['accuracy']:6.2f}  "
              f"precision {100 * m['precision']:6.2f}  recall {100 * m['recall']:6.2f}  "
              f"F1 {100 * m['f1']:6.2f}   TP {m['TP']} TN {m['TN']} FP {m['FP']} FN {m['FN']}")
    print(f"[saved] predictions in {out_dir}")


# ---------------------------------------------------------------------------
# Mode 2: ensemble analysis
# ---------------------------------------------------------------------------
def image_key(path: str) -> str:
    """'<class folder>/<file name>', independent of the data root and of / vs \\."""
    return "/".join([p for p in re.split(r"[\\/]+", str(path)) if p][-2:])


def load_members(run_dirs, split: str):
    """Labels and P [members, images]; every member's rows matched to the first by image."""
    ref_keys = ref_labels = None
    P = []
    for d in run_dirs:
        f = Path(d) / f"{split}_predictions.npz"
        if not f.exists():
            raise SystemExit(f"Missing {f}. Run test.py --ckpt for this network first.")
        z = np.load(f, allow_pickle=True)
        keys = np.array([image_key(p) for p in z["paths"]])
        labels, scores = z["labels"].astype(int), z["scores"].astype(float)
        if len(set(keys)) != len(keys):
            raise SystemExit(f"{f}: duplicate images.")
        if ref_keys is None:
            ref_keys, ref_labels = keys, labels
            P.append(scores)
            continue
        if set(keys) != set(ref_keys):
            raise SystemExit(f"{f} does not contain the same images as {run_dirs[0]}.")
        pos = {k: i for i, k in enumerate(keys)}
        idx = np.array([pos[k] for k in ref_keys])
        if not np.array_equal(labels[idx], ref_labels):
            raise SystemExit(f"{f}: labels differ from {run_dirs[0]}.")
        P.append(scores[idx])
    return ref_labels, np.stack(P)


def kept_count(n: int, rate: float) -> int:
    """Tablets classified automatically: (1 - rate) x n, rounded to the nearest tablet."""
    return int(round((1.0 - rate) * n))


def choose_deferral(labels, P) -> float:
    err = errors(labels, P.mean(axis=0))
    order = np.argsort(entropy(P.mean(axis=0)), kind="stable")    # least uncertain first
    best_rate, best_acc = None, -1.0
    for rate in DEFER_GRID:
        acc = 1.0 - err[order[:kept_count(len(labels), rate)]].mean()
        if acc > best_acc + 1e-12:                                 # ties keep the smaller rate
            best_rate, best_acc = rate, acc
    return best_rate


def auroc_gain(err_ens, H, member_scores, member_errs=None):
    """Observed gain (ensemble AUROC - mean member AUROC) and its paired bootstrap 95% interval.

    member_errs=None: same-target comparison, every score ranks the ensemble's errors.
    member_errs given: own-error comparison, each member ranks its own errors.
    """
    targets = [err_ens] * len(member_scores) if member_errs is None else member_errs
    observed = auroc(err_ens, H) - np.mean([auroc(t, s) for t, s in zip(targets, member_scores)])
    rng = np.random.default_rng(BOOT_SEED)
    n, diffs = len(err_ens), []
    for _ in range(BOOT_N):
        idx = rng.integers(0, n, n)
        e = err_ens[idx]
        tt = [t[idx] for t in targets]
        if any(t.min() == t.max() for t in [e, *tt]):
            continue
        diffs.append(auroc(e, H[idx]) - np.mean([auroc(t, s[idx]) for t, s in zip(tt, member_scores)]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(observed), float(lo), float(hi)


def analyse(title, labels, P, rate, out):
    n = len(labels)
    pbar = P.mean(axis=0)
    err = errors(labels, pbar)
    H = entropy(pbar)
    MI = H - entropy(P).mean(axis=0)
    out.append(f"== {title}  (n = {n}, defective {int(labels.sum())}) ==")

    m_scores, m_errs, own, rows = [], [], [], []
    for i in range(len(P)):
        m = metrics(labels, P[i])
        e = errors(labels, P[i])
        s = np.minimum(P[i], 1 - P[i])                  # 1 - max(p, 1 - p)
        m_scores.append(s)
        m_errs.append(e)
        own.append(auroc(e, s))
        rows.append(m)
        out.append(f"  network {i + 1}: accuracy {100 * m['accuracy']:6.2f}  precision {100 * m['precision']:6.2f}  "
                   f"recall {100 * m['recall']:6.2f}  F1 {100 * m['f1']:6.2f}  errors {int(e.sum())}  "
                   f"error AUROC (own errors) {own[-1]:.3f}")
    for k in ("accuracy", "precision", "recall", "f1"):
        v = 100 * np.array([r[k] for r in rows])
        out.append(f"  mean +/- SD {k:<9} {v.mean():6.2f} +/- {v.std(ddof=1):.2f}")

    m = metrics(labels, pbar)
    ci = {"accuracy": wilson(m["TP"] + m["TN"], n), "precision": wilson(m["TP"], m["TP"] + m["FP"]),
          "recall": wilson(m["TP"], m["TP"] + m["FN"])}
    out.append(f"  ensemble: accuracy {100 * m['accuracy']:.2f}  precision {100 * m['precision']:.2f}  "
               f"recall {100 * m['recall']:.2f}  F1 {100 * m['f1']:.2f}  "
               f"(TP {m['TP']}, TN {m['TN']}, FP {m['FP']}, FN {m['FN']}; errors {int(err.sum())})")
    out.append("  Wilson 95% intervals: " + ", ".join(
        f"{k} {100 * lo:.1f}-{100 * hi:.1f}" for k, (lo, hi) in ci.items()))

    out.append(f"  error AUROC: ensemble entropy {auroc(err, H):.3f}, mutual information {auroc(err, MI):.3f}, "
               f"mean member on own errors {np.mean(own):.3f}, "
               f"mean member on ensemble errors {np.mean([auroc(err, s) for s in m_scores]):.3f}")
    for label, kw in (("same-target (ensemble errors)", {}), ("own-error", {"member_errs": m_errs})):
        g, lo, hi = auroc_gain(err, H, m_scores, **kw)
        out.append(f"  AUROC gain, {label}: {g:+.3f}  (95% bootstrap interval {lo:+.3f} to {hi:+.3f})")

    order = np.argsort(H, kind="stable")
    pred = (pbar >= THRESHOLD).astype(int)
    out.append(f"  selective review (deferral rate chosen on validation: {int(round(100 * rate))}%)")
    out.append("    rate  kept  deferred  accuracy kept  errors deferred  kept: FN  FP  defective kept")
    for r in DEFER_TABLE:
        k = kept_count(n, r)
        kept, deferred = order[:k], order[k:]
        c = confusion(labels[kept], pred[kept])
        mark = "  <- selected" if abs(r - rate) < 1e-9 else ""
        out.append(f"    {int(round(100 * r)):>3}%  {k:4d}  {n - k:8d}  {100 * (1 - err[kept].mean()):12.2f}  "
                   f"{int(err[deferred].sum()):15d}  {c['FN']:8d}  {c['FP']:2d}  {int(labels[kept].sum()):13d}{mark}")

    out.append("  ensembles of 1, 2 and 3 members (mean over all subsets; 20% deferral)")
    out.append("    members  accuracy     F1  errors  AUROC  accuracy kept")
    for size in range(1, len(P) + 1):
        res = []
        for sub in itertools.combinations(range(len(P)), size):
            pb = P[list(sub)].mean(axis=0)
            e = errors(labels, pb)
            h = entropy(pb)
            keep = np.argsort(h, kind="stable")[:kept_count(n, 0.2)]
            mm = metrics(labels, pb)
            res.append((mm["accuracy"], mm["f1"], e.sum(), auroc(e, h), 1 - e[keep].mean()))
        r = np.nanmean(np.array(res, dtype=float), axis=0)
        out.append(f"    {size:7d}  {100 * r[0]:8.2f}  {100 * r[1]:5.2f}  {r[2]:6.1f}  {r[3]:.3f}  {100 * r[4]:13.2f}")
    out.append("")


def ensemble(a) -> None:
    out = [f"UAMSF-Net ensemble of {len(a.ensemble)} networks; decision threshold p >= {THRESHOLD}", ""]
    vl, VP = load_members(a.ensemble, "val")
    rate = choose_deferral(vl, VP)
    out.append(f"Validation: ensemble errors {int(errors(vl, VP.mean(axis=0)).sum())} of {len(vl)}; "
               f"chosen deferral rate {int(round(100 * rate))}%")
    out.append("")
    analyse("PV test set", *load_members(a.ensemble, "test"), rate, out)
    if all((Path(d) / "pillqc_predictions.npz").exists() for d in a.ensemble):
        analyse("pillQC (no fine-tuning)", *load_members(a.ensemble, "pillqc"), rate, out)
    text = "\n".join(out)
    print(text)
    Path("ensemble_results.txt").write_text(text + "\n", encoding="utf-8")
    print("[saved] ensemble_results.txt")


# ---------------------------------------------------------------------------
# Mode 3: latency
# ---------------------------------------------------------------------------
@torch.no_grad()
def latency(a) -> None:
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models = [load_model(c, device) for c in a.latency]
    x = torch.randn(1, 3, 224, 224, device=device)

    def timed(fn, warmup=50, runs=500):
        for _ in range(warmup):
            fn()
        ts = []
        for _ in range(runs):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t = time.perf_counter()
            fn()
            if device.type == "cuda":
                torch.cuda.synchronize()
            ts.append(1000 * (time.perf_counter() - t))
        q1, med, q3 = np.percentile(ts, [25, 50, 75])
        return med, q3 - q1

    single = timed(lambda: models[0](x)["probs"])
    ens = timed(lambda: torch.stack([m(x)["probs"] for m in models]).mean(0))
    for name, (med, iqr) in (("one network", single), (f"ensemble of {len(models)}", ens)):
        print(f"{name:<15} median {med:.2f} ms (IQR {iqr:.2f}), {1000 / med:.2f} images per second")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", help="test one network")
    ap.add_argument("--data", default="data/PV_augmented")
    ap.add_argument("--pillqc", default=None, help="pillQC images folder")
    ap.add_argument("--ensemble", nargs="+", help="run folders of the ensemble members")
    ap.add_argument("--latency", nargs="+", help="checkpoints to time")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.ensemble:
        ensemble(a)
    elif a.latency:
        latency(a)
    elif a.ckpt:
        test_one(a)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
