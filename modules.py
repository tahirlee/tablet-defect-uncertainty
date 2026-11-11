"""
Architectural modules: AMSM, DAM (ablation only), and the evidential head.

AMSM follows Josh et al. 2026 Sect. 3.3.2 and Appendix A.1:
  - three depthwise-separable dilated branches at rates {1, 2, 4}
  - one identity branch
  - four-branch SCALAR softmax gating (one weight per branch, not per channel),
    computed per sample from GAP descriptors of the four branches
  - output is a convex combination, so feature magnitude stays stable

One correction to the source paper: its text says the gate MLP hidden size is
max(C/4, 16), but its own parameter table (Table 3: 2368 / 18670 / 147940 for
C = 48 / 136 / 384) is only reproducible with hidden = C // 4. We use C // 4,
which reproduces their reported 692,490 AMSM parameters exactly.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_norm(channels: int, style: str = "bn") -> nn.Module:
    """Normalization for the AMSM branches.

    'bn' reproduces the source paper. 'gn' (GroupNorm) is batch-size
    independent, which matters here: the PV protocol fixes batch size at 16 on
    a 1,140-image training set, where BatchNorm running statistics are noisy.
    Use 'gn' with LayerNorm-based backbones such as ConvNeXt so the whole
    network is free of batch-coupled statistics.
    """
    if style == "bn":
        return nn.BatchNorm2d(channels)
    if style == "gn":
        groups = 32
        while groups > 1 and channels % groups != 0:
            groups //= 2
        return nn.GroupNorm(groups, channels)
    raise ValueError(f"unknown norm style '{style}'")


class DepthwiseSeparableDilated(nn.Module):
    """m_r = ReLU(Norm(PW(DW_r(x)))), padding preserves spatial size."""

    def __init__(self, channels: int, dilation: int, kernel_size: int = 3,
                 norm: str = "bn"):
        super().__init__()
        pad = dilation * (kernel_size - 1) // 2
        self.dw = nn.Conv2d(
            channels, channels, kernel_size,
            padding=pad, dilation=dilation, groups=channels, bias=False,
        )
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = make_norm(channels, norm)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class AMSM(nn.Module):
    """Adaptive Multi-Scale Module.

    Returns the fused feature map and the four gate weights. The gate weights
    are a by-product of the fusion; nothing in this project interprets them.
    """

    def __init__(
        self,
        channels: int,
        dilations: Sequence[int] = (1, 2, 4),
        gate_hidden_div: int = 4,
        norm: str = "bn",
    ):
        super().__init__()
        self.channels = channels
        self.dilations = tuple(dilations)
        self.n_branches = len(self.dilations) + 1  # + identity

        self.branches = nn.ModuleList(
            [DepthwiseSeparableDilated(channels, d, norm=norm) for d in self.dilations]
        )

        hidden = max(channels // gate_hidden_div, 4)
        self.gate = nn.Sequential(
            nn.Linear(self.n_branches * channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, self.n_branches),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = [b(x) for b in self.branches] + [x]          # identity last
        desc = torch.cat([o.mean(dim=(2, 3)) for o in outs], dim=1)  # [B, 4C]
        w = F.softmax(self.gate(desc), dim=1)                # [B, 4], sums to 1

        stacked = torch.stack(outs, dim=1)                   # [B, 4, C, H, W]
        fused = (stacked * w[:, :, None, None, None]).sum(dim=1)
        return fused, w

    def branch_names(self) -> list:
        return [f"dilation_{d}" for d in self.dilations] + ["identity"]


class StageProjection(nn.Module):
    """Optional 1x1 projection applied BEFORE AMSM.

    AMSM cost is quadratic in channel width (the pointwise conv is C^2 per
    branch, and the gate MLP is 4C x C/4). A backbone whose deepest stage is
    768 or 1280 channels makes AMSM larger than it needs to be. Capping each
    stage at `out_channels` keeps the module affordable and keeps the
    concatenated descriptor a sensible width, at the cost of a small amount of
    backbone capacity. Set out_channels = None to disable.
    """

    def __init__(self, in_channels: int, out_channels: int, norm: str = "bn"):
        super().__init__()
        if in_channels == out_channels:
            self.block = nn.Identity()
            self.out_channels = in_channels
            return
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# DAM -- implemented for the ablation only. Excluded from the deployed model.
# ---------------------------------------------------------------------------
class ECA(nn.Module):
    """Efficient Channel Attention: 1D conv over channels, adaptive kernel."""

    def __init__(self, channels: int, gamma: int = 2, b: int = 1):
        super().__init__()
        import math

        t = int(abs((math.log2(channels) + b) / gamma))
        k = t if t % 2 else t + 1
        self.conv = nn.Conv1d(1, 1, kernel_size=max(k, 3), padding=max(k, 3) // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=(2, 3)).unsqueeze(1)          # [B, 1, C]
        y = torch.sigmoid(self.conv(y)).transpose(1, 2).unsqueeze(-1)
        return x * y


class SpatialAttention(nn.Module):
    """7x7 conv over channel-concatenated average and max projections."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))


class DAM(nn.Module):
    """Dual Attention Module (ECA then spatial).

    Josh et al. found this degrades CROSS-DATASET accuracy when stacked after
    AMSM, despite improving in-domain calibration, and excluded it. It is kept
    here so the four-variant ablation can be rerun on PV / pillQC, where the
    same question is worth asking independently.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.eca = ECA(channels)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.eca(x))


# ---------------------------------------------------------------------------
# Classification heads
# ---------------------------------------------------------------------------
class MLPHead(nn.Module):
    """Shared trunk: in -> 256 (ReLU, drop .3) -> 128 (ReLU, drop .2) -> K."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int = 2,
        hidden: Sequence[int] = (256, 128),
        dropout: Sequence[float] = (0.3, 0.2),
    ):
        super().__init__()
        layers = []
        d = in_dim
        for h, p in zip(hidden, dropout):
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True), nn.Dropout(p)]
            d = h
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def evidential_outputs(logits: torch.Tensor, num_classes: int) -> dict:
    """Dirichlet parameterization (Josh et al. Sect. 3.3.4).

    evidence e = softplus(z);  alpha = e + 1;  S = sum(alpha)
    p = alpha / S;  vacuity u = K / S  in (0, 1]

    u is the single-pass uncertainty score. Unlike the composite TRI of the
    trust-graph literature, u is bounded by construction and its direction is
    fixed: larger S means more accumulated evidence and therefore lower u.
    """
    evidence = F.softplus(logits)
    alpha = evidence + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    probs = alpha / S
    u = num_classes / S.squeeze(1)
    return {
        "evidence": evidence,
        "alpha": alpha,
        "strength": S.squeeze(1),
        "probs": probs,
        "uncertainty": u,
    }
