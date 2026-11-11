"""
Training objectives.

THREE are provided, and all three should be reported. Together they decompose
any gain into architecture / focal / Dirichlet contributions, which is the only
honest way to claim an improvement over a benchmark whose protocol fixes
the loss.

  PlainCELoss ('ce')     -- standard cross-entropy, nothing else. This is what
      the PV protocol specifies for every benchmarked architecture (Sect. 2.3:
      "standard cross-entropy (PyTorch nn.CrossEntropyLoss with default
      parameters: no class weighting, no label smoothing, no logit
      modification), applied identically to every architecture"). Training the
      proposed architecture under this loss is the only strictly like-for-like
      comparison against the twelve baselines, and it isolates the
      ARCHITECTURE's contribution. **This is the row for the main table.**

  CEFocalLoss ('cefocal') -- cross-entropy + focal correction. This is what the
      source paper calls "FairCELoss" and what it trained its own six baselines
      under (Josh et al. Sect. 3.4). Reporting it separates the focal term's
      effect from the Dirichlet parameterization's, which otherwise arrive
      together in the EDL row and cannot be told apart.

  EvidentialLoss ('edl')  -- L_EDL + lambda_KL(t) * L_KL + lambda_FOCAL *
      L_focal. The full proposed system. It departs from the PV loss
      specification, so it is reported as an additional row, never substituted
      for the CE row.

Naming note: the source paper's "FairCELoss" already contains a focal term, so
it is not the PV protocol's loss despite the name. Do not use it as the
protocol-compliant comparison row; use 'ce'.

Note on class weighting: PV splits are exactly 1:1 (190/190, 30/30, 100/100),
so inverse-frequency weighting is a no-op here and is deliberately omitted.
The asymmetric cost of false negatives is handled by the PV protocol's own
mechanism -- F2 checkpoint selection at a 0.40 decision threshold -- not by
reweighting the loss.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import LOSS, LossConfig


def _one_hot(targets: torch.Tensor, num_classes: int) -> torch.Tensor:
    return F.one_hot(targets, num_classes).float()


def _focal_term(probs: torch.Tensor, y_onehot: torch.Tensor,
                weights: torch.Tensor, gamma: float) -> torch.Tensor:
    """Class-weighted focal correction, shared by CEFocalLoss and EvidentialLoss.

    L = -sum_k w_k * y_k * (1 - p_k)^gamma * log(p_k)
    """
    p = probs.clamp_min(1e-7)
    w = weights.to(p.device).unsqueeze(0)
    return -(w * y_onehot * (1.0 - p).pow(gamma) * torch.log(p)).sum(dim=1).mean()


class PlainCELoss(nn.Module):
    """Standard cross-entropy. PV-protocol compliant -- the main comparison row."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self.ce = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int = 0) -> dict:
        loss = self.ce(outputs["logits"], targets)
        return {"loss": loss, "ce": loss.detach()}


class CEFocalLoss(nn.Module):
    """Cross-entropy + focal correction.

    This is the source paper's "FairCELoss". It is NOT the PV protocol's loss --
    the focal term is an addition. Its purpose here is to sit between the plain
    CE row and the full EDL row so the focal and Dirichlet contributions can be
    attributed separately.
    """

    def __init__(self, num_classes: int = 2, cfg: LossConfig = LOSS):
        super().__init__()
        self.K = num_classes
        self.cfg = cfg
        self.ce = nn.CrossEntropyLoss()
        w = torch.tensor(cfg.class_weights[:num_classes], dtype=torch.float32)
        self.register_buffer("class_weights", w)

    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int = 0) -> dict:
        logits = outputs["logits"]
        probs = outputs.get("probs")
        if probs is None:
            probs = torch.softmax(logits, dim=1)

        l_ce = self.ce(logits, targets)
        l_focal = _focal_term(probs, _one_hot(targets, self.K),
                              self.class_weights, self.cfg.focal_gamma)
        loss = l_ce + self.cfg.lambda_focal * l_focal
        return {"loss": loss, "ce": l_ce.detach(), "focal": l_focal.detach()}


# Backwards-compatible alias. Prefer the explicit names above: the source
# paper's "FairCELoss" includes a focal term, which the PV protocol does not.
FairCELoss = PlainCELoss


class EvidentialLoss(nn.Module):
    """Dirichlet evidential objective with KL annealing and a focal correction."""

    def __init__(self, num_classes: int = 2, cfg: LossConfig = LOSS):
        super().__init__()
        self.K = num_classes
        self.cfg = cfg
        w = torch.tensor(cfg.class_weights[:num_classes], dtype=torch.float32)
        self.register_buffer("class_weights", w)

    # -- components ---------------------------------------------------------
    @staticmethod
    def _edl_digamma(alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """L_EDL = sum_k y_k [psi(S) - psi(alpha_k)]  (Josh et al. Eq. 5)."""
        S = alpha.sum(dim=1, keepdim=True)
        return (y * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1).mean()

    def _kl_to_uniform(self, alpha: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """KL(Dir(alpha_tilde) || Dir(1)) on the non-evidence entries.

        alpha_tilde removes the evidence supporting the true class, so the term
        only penalises evidence assigned to WRONG classes.
        """
        a = y + (1.0 - y) * alpha
        K = self.K
        S = a.sum(dim=1, keepdim=True)
        ones = torch.ones_like(a)

        term1 = torch.lgamma(S.squeeze(1)) - torch.lgamma(torch.tensor(float(K), device=a.device))
        term1 = term1 - torch.lgamma(a).sum(dim=1)
        term2 = ((a - ones) * (torch.digamma(a) - torch.digamma(S))).sum(dim=1)
        return (term1 + term2).mean()

    def _focal(self, probs: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Focal correction on the Dirichlet mean probabilities (Eq. 7)."""
        return _focal_term(probs, y, self.class_weights, self.cfg.focal_gamma)

    # -- forward ------------------------------------------------------------
    def forward(self, outputs: dict, targets: torch.Tensor, epoch: int = 0) -> dict:
        alpha = outputs["alpha"]
        probs = outputs["probs"]
        y = _one_hot(targets, self.K)

        l_edl = self._edl_digamma(alpha, y)
        l_kl = self._kl_to_uniform(alpha, y)
        l_focal = self._focal(probs, y)

        # lambda_KL(t) = min(1, t / anneal_epochs) * lambda_KL  (Eq. 6).
        # Annealing prevents premature collapse of the Dirichlet evidence.
        anneal = min(1.0, (epoch + 1) / max(self.cfg.kl_anneal_epochs, 1))
        lam_kl = anneal * self.cfg.lambda_kl

        loss = l_edl + lam_kl * l_kl + self.cfg.lambda_focal * l_focal
        return {
            "loss": loss,
            "edl": l_edl.detach(),
            "kl": l_kl.detach(),
            "focal": l_focal.detach(),
            "lambda_kl": torch.tensor(lam_kl),
        }


LOSSES = {
    "ce": "plain cross-entropy (PV protocol -- main comparison row)",
    "cefocal": "cross-entropy + focal (the source paper's 'FairCELoss')",
    "edl": "evidential: L_EDL + lambda_KL(t)*L_KL + lambda_FOCAL*L_focal",
}


def build_loss(head_type: str, num_classes: int = 2, cfg: LossConfig = LOSS) -> nn.Module:
    """`head_type` is the --head flag: 'ce', 'cefocal' or 'edl'."""
    if head_type == "edl":
        return EvidentialLoss(num_classes, cfg)
    if head_type == "cefocal":
        return CEFocalLoss(num_classes, cfg)
    if head_type == "ce":
        return PlainCELoss(num_classes)
    raise ValueError(f"unknown loss '{head_type}'; choose from {list(LOSSES)}")
