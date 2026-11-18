"""
UAMSF-Net: ConvNeXt-Tiny backbone + multi-scale module (MSM) + MLP head.

    image 224x224
      -> ConvNeXt-Tiny (ImageNet-1K weights, torchvision); feature maps of
         stages 2, 3 and 4 (192, 384 and 768 channels; strides 8, 16, 32)
      -> one MSM per stage
      -> global average pooling, concatenation (1344 values)
      -> MLP 1344 -> 256 -> 128 -> 2

Output index 1 is the defective class. The model has 31,338,686 parameters:
backbone 27,818,592, three MSMs 3,142,620, head 377,474.

    python model.py      # builds the model and prints the parameter counts
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

STAGE_CHANNELS = (192, 384, 768)
STAGE_INDICES = (3, 5, 7)      # blocks of torchvision's convnext_tiny().features
DILATIONS = (1, 2, 4)


def group_norm(channels: int) -> nn.GroupNorm:
    groups = 32
    while groups > 1 and channels % groups != 0:
        groups //= 2
    return nn.GroupNorm(groups, channels)


class DilatedBranch(nn.Module):
    """M_r = ReLU(GN(PW(DW_r(F)))): depthwise 3x3 with dilation r, then pointwise 1x1."""

    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.dw = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation,
                            groups=channels, bias=False)
        self.pw = nn.Conv2d(channels, channels, 1, bias=False)
        self.norm = group_norm(channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pw(self.dw(x))))


class MSM(nn.Module):
    """Multi-scale module (paper Sect. 2.1.3, Eqs. 1-4).

    Three dilated branches (r = 1, 2, 4) and an identity branch, combined with
    per-image softmax weights from a two-layer gate (4C -> C/4 -> 4).
    """

    def __init__(self, channels: int, dilations: Sequence[int] = DILATIONS):
        super().__init__()
        self.branches = nn.ModuleList([DilatedBranch(channels, d) for d in dilations])
        n = len(dilations) + 1
        hidden = max(channels // 4, 4)
        self.gate = nn.Sequential(nn.Linear(n * channels, hidden), nn.ReLU(inplace=True),
                                  nn.Linear(hidden, n))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outs = [b(x) for b in self.branches] + [x]                       # identity last
        d = torch.cat([o.mean(dim=(2, 3)) for o in outs], dim=1)         # Eq. (2)
        w = F.softmax(self.gate(d), dim=1)                               # Eq. (3)
        fused = (torch.stack(outs, dim=1) * w[:, :, None, None, None]).sum(dim=1)  # Eq. (4)
        return fused, w


class Head(nn.Module):
    def __init__(self, in_dim: int = sum(STAGE_CHANNELS), num_classes: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Backbone(nn.Module):
    """ConvNeXt-Tiny up to stage 4, returning the maps of stages 2, 3 and 4."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        import torchvision.models as M
        weights = M.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        self.features = M.convnext_tiny(weights=weights).features   # stochastic depth 0.1
        # Check the stage widths once with a dummy image (as in the training runs of the paper;
        # this also keeps the random-number sequence, and hence seeded initialisation, identical).
        with torch.no_grad():
            x, widths = torch.zeros(1, 3, 224, 224), {}
            for i, block in enumerate(self.features):
                x = block(x)
                widths[i] = x.shape[1]
        if tuple(widths[i] for i in STAGE_INDICES) != STAGE_CHANNELS:
            raise RuntimeError("unexpected ConvNeXt-Tiny layout; check the torchvision version")

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        maps = []
        for i, block in enumerate(self.features):
            x = block(x)
            if i in STAGE_INDICES:
                maps.append(x)
        return maps


class UAMSFNet(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = Backbone(pretrained)
        # The attribute is named "amsm" so that saved checkpoints load unchanged.
        self.amsm = nn.ModuleList([MSM(c) for c in STAGE_CHANNELS])
        self.head = Head()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        pooled = []
        for f, msm in zip(self.backbone(x), self.amsm):
            f, _ = msm(f)
            pooled.append(f.mean(dim=(2, 3)))
        logits = self.head(torch.cat(pooled, dim=1))
        return {"logits": logits, "probs": torch.softmax(logits, dim=1)}

    def param_counts(self) -> Dict[str, int]:
        n = lambda m: sum(p.numel() for p in m.parameters())
        return {"backbone": n(self.backbone), "msm": n(self.amsm), "head": n(self.head),
                "total": n(self)}


def load_model(ckpt_path: str, device: torch.device) -> UAMSFNet:
    model = UAMSFNet(pretrained=False)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    return model.to(device).eval()


if __name__ == "__main__":
    m = UAMSFNet(pretrained=False)
    for k, v in m.param_counts().items():
        print(f"{k:<9}{v:>12,}")
    out = m(torch.randn(2, 3, 224, 224))
    print("probs", tuple(out["probs"].shape))
