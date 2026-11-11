"""
Verifies the AMSM parameter arithmetic against Josh et al. Table 3, and
resolves a contradiction in the source paper. Pure arithmetic -- no torch.

The paper's TEXT (Sect. 3.3.2) states the gate MLP hidden size is
max(C/4, 16). Its own TABLE 3 reports gate-MLP parameter counts of
2368 / 18670 / 147940 for C = 48 / 136 / 384, totalling 692,490 AMSM
parameters. These two statements are inconsistent: at C = 48, max(48/4, 16)
= 16, which yields 3156 parameters, not 2368. A hidden size of C // 4 = 12
yields exactly 2368.

modules.py therefore implements hidden = C // 4 (with a floor of 4, which
never binds for C >= 16). This script shows that choice reproduces every
published figure exactly and the paper's stated rule does not.

    python verify_amsm_params.py
"""
from __future__ import annotations

# Josh et al. Table 3: AMSM parameter breakdown by stage (EfficientNet-B3).
PAPER_TABLE_3 = {
    #  C: (DW, PW, BN, branch_total, gate_MLP, stage_total)
    48:  (432,  2304,   96,  2832,   2368,  10864),
    136: (1224, 18496,  272, 19992,  18670, 78646),
    384: (3456, 147456, 768, 151680, 147940, 602980),
}
PAPER_AMSM_TOTAL = 692_490
PAPER_EQUIV_STANDARD_CONV = 4_546_320
PAPER_SAVINGS_RATIO = 6.6


def branch_params(c: int, k: int = 3) -> dict:
    """One depthwise-separable dilated branch: DW(k x k) -> PW(1x1) -> BN.

    Convolutions are bias-free; BatchNorm contributes 2C (weight + bias).
    Dilation changes the receptive field, not the parameter count.
    """
    dw = c * k * k
    pw = c * c
    bn = 2 * c
    return {"dw": dw, "pw": pw, "bn": bn, "total": dw + pw + bn}


def gate_params(c: int, hidden: int, n_branches: int = 4) -> int:
    """Two-layer MLP: Linear(n*C -> hidden) -> ReLU -> Linear(hidden -> n)."""
    l1 = (n_branches * c) * hidden + hidden
    l2 = hidden * n_branches + n_branches
    return l1 + l2


def equivalent_standard_conv(c: int, k: int = 3, n_dilated: int = 3) -> int:
    """What the same branches would cost as dense C->C k x k convolutions.

    The comparison replaces only the depthwise+pointwise factorization; the
    BatchNorm after each branch is unchanged and so is carried over. Dropping
    it undercounts by 3 * 2C per stage (3,408 across the three stages), which
    is exactly the gap against the paper's published 4,546,320.
    """
    return n_dilated * (c * c * k * k + 2 * c)


def main() -> None:
    print(__doc__)
    print("=" * 74)
    print(f"{'C':>5} {'DW':>8} {'PW':>9} {'BN':>6} {'branch':>9} "
          f"{'gate(C//4)':>11} {'stage':>9}  {'paper':>9}  match")
    print("-" * 74)

    total_ours = 0
    all_ok = True
    for c, (p_dw, p_pw, p_bn, p_branch, p_gate, p_stage) in PAPER_TABLE_3.items():
        b = branch_params(c)
        hidden = max(c // 4, 4)
        g = gate_params(c, hidden)
        stage = 3 * b["total"] + g
        total_ours += stage

        ok = (b["dw"], b["pw"], b["bn"], b["total"], g, stage) == \
             (p_dw, p_pw, p_bn, p_branch, p_gate, p_stage)
        all_ok &= ok
        print(f"{c:>5} {b['dw']:>8} {b['pw']:>9} {b['bn']:>6} {b['total']:>9} "
              f"{g:>11} {stage:>9}  {p_stage:>9}  {'ok' if ok else 'MISMATCH'}")

    print("-" * 74)
    print(f"{'total':>5} {'':>8} {'':>9} {'':>6} {'':>9} {'':>11} "
          f"{total_ours:>9}  {PAPER_AMSM_TOTAL:>9}  "
          f"{'ok' if total_ours == PAPER_AMSM_TOTAL else 'MISMATCH'}")
    all_ok &= total_ours == PAPER_AMSM_TOTAL

    # --- the contradiction, made explicit --------------------------------
    print("\n" + "=" * 74)
    print("Gate MLP: the paper's stated rule vs the rule that fits its own table")
    print("=" * 74)
    print(f"{'C':>5} {'hidden=C//4':>12} {'params':>8}   "
          f"{'hidden=max(C/4,16)':>19} {'params':>8}   {'paper':>8}")
    print("-" * 74)
    text_rule_ok = True
    for c, (*_, p_gate, _) in PAPER_TABLE_3.items():
        h_floor = max(c // 4, 4)
        h_text = max(c // 4, 16)
        g_floor, g_text = gate_params(c, h_floor), gate_params(c, h_text)
        text_rule_ok &= (g_text == p_gate)
        mark = "  <- matches" if g_floor == p_gate else ""
        print(f"{c:>5} {h_floor:>12} {g_floor:>8}   {h_text:>19} "
              f"{g_text:>8}   {p_gate:>8}{mark}")
    print("-" * 74)
    print(f"  C // 4            reproduces Table 3: {all_ok}")
    print(f"  max(C/4, 16)      reproduces Table 3: {text_rule_ok}")
    print("  -> modules.py implements C // 4, matching the paper's numbers "
          "rather than its prose.")

    # --- the efficiency claim ---------------------------------------------
    print("\n" + "=" * 74)
    print("Depthwise-separable savings claim (paper: 4,546,320 -> 692,490, ~6.6x)")
    print("=" * 74)
    equiv = sum(equivalent_standard_conv(c) for c in PAPER_TABLE_3)
    ratio = equiv / total_ours
    print(f"  equivalent standard 3x3 convolutions : {equiv:>10,}  "
          f"(paper {PAPER_EQUIV_STANDARD_CONV:,})")
    print(f"  AMSM                                 : {total_ours:>10,}  "
          f"(paper {PAPER_AMSM_TOTAL:,})")
    print(f"  savings ratio                        : {ratio:>10.2f}x "
          f"(paper ~{PAPER_SAVINGS_RATIO}x)")
    equiv_ok = equiv == PAPER_EQUIV_STANDARD_CONV
    ratio_ok = abs(ratio - PAPER_SAVINGS_RATIO) < 0.05
    all_ok &= equiv_ok and ratio_ok
    print(f"  equivalent-conv count matches: {equiv_ok}; "
          f"ratio matches: {ratio_ok}")

    # --- non-backbone budget ----------------------------------------------
    print("\n" + "=" * 74)
    print("Non-backbone cost under the <5M budget (EfficientNet-B0 default)")
    print("=" * 74)
    print(f"{'proj':>6}{'stage channels':>22}{'proj':>10}{'AMSM':>10}"
          f"{'head':>9}{'non-backbone':>14}{'backbone left':>15}")
    print("-" * 74)

    CAP = 5_000_000
    stage_in = (40, 112, 320)          # EfficientNet-B0 at strides 8/16/32
    for proj in (64, 96, 128, 192, 256):
        chans = tuple(min(c, proj) for c in stage_in)
        p_proj = sum(
            0 if ci == co else ci * co + 2 * co      # 1x1 conv + GroupNorm
            for ci, co in zip(stage_in, chans)
        )
        p_amsm = sum(3 * branch_params(c)["total"] + gate_params(c, max(c // 4, 4))
                     for c in chans)
        feat = sum(chans)
        p_head = (feat * 256 + 256) + (256 * 128 + 128) + (128 * 2 + 2)
        non_bb = p_proj + p_amsm + p_head
        print(f"{proj:>6}{str(chans):>22}{p_proj:>10,}{p_amsm:>10,}"
              f"{p_head:>9,}{non_bb:>14,}{CAP - non_bb:>15,}")

    print("-" * 74)
    print("""  AMSM cost grows as C^2, so the projection width is the main lever.
  At the default proj=128 the non-backbone parts cost 278,020 parameters,
  leaving ~4.72M for the encoder -- comfortably more than EfficientNet-B0
  needs (~3.6M once its unused 1280-channel head block is truncated).

  Run `python param_budget.py` for the real totals with torch, which also
  prints PV's twelve baselines for comparison.""")

    print("\n" + "=" * 74)
    if not all_ok:
        raise SystemExit("  FAILED: AMSM arithmetic does not match Table 3")
    print("  PASS: AMSM parameterization is exact against Josh et al. Table 3")
    print("=" * 74)


if __name__ == "__main__":
    main()
