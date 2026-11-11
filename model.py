"""
TabletUNet-E: uncertainty-aware tablet defect classifier.

  ImageNet-pretrained backbone (ConvNeXt-Tiny by default)
    -> optional 1x1 stage projection (caps AMSM cost)
    -> AMSM at each of three stages (strides 8 / 16 / 32)
    -> global average pool + concatenate
    -> evidential (Dirichlet) head, K = 2

Binary task: index 0 = normal, index 1 = defective (the positive class, per the
PV protocol). The evidential head yields a single-pass vacuity score u = 2 / S
that is bounded in (0, 1] and used for misclassification detection, selective
prediction, and the uncertainty CAM.

The backbone is pluggable (see backbones.py). Swapping it does not change
anything downstream.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from backbones import MultiStageBackbone, REGISTRY, describe_backbones
from config import MODEL, ModelConfig
from modules import AMSM, DAM, MLPHead, StageProjection, evidential_outputs


class TabletDefectNet(nn.Module):
    def __init__(self, cfg: ModelConfig = MODEL):
        super().__init__()
        self.cfg = cfg

        self.backbone = MultiStageBackbone(
            cfg.backbone, cfg.feature_strides, cfg.pretrained
        )
        chans = list(self.backbone.out_channels)

        # AMSM normalization defaults to match the backbone's own style, so a
        # LayerNorm backbone is not silently paired with batch-coupled BN at
        # batch size 16.
        norm = cfg.amsm_norm or ("gn" if self.backbone.spec.norm_style == "ln" else "bn")
        self.amsm_norm = norm

        # Optional channel cap before AMSM.
        if cfg.proj_channels:
            self.proj = nn.ModuleList(
                [StageProjection(c, min(c, cfg.proj_channels), norm) for c in chans]
            )
            chans = [p.out_channels for p in self.proj]
        else:
            self.proj = None

        self.amsm = nn.ModuleList(
            [AMSM(c, cfg.amsm_dilations, cfg.amsm_gate_hidden_div, norm) for c in chans]
        ) if cfg.use_amsm else None

        self.dam = nn.ModuleList([DAM(c) for c in chans]) if cfg.use_dam else None

        self.stage_channels = chans
        self.feat_dim = sum(chans)
        self.head = MLPHead(
            self.feat_dim, cfg.num_classes, cfg.head_hidden, cfg.head_dropout
        )

        self._last_maps: List[torch.Tensor] = []
        self._check_budget()

    def _check_budget(self) -> None:
        """Fail at construction, not an hour into training."""
        cap = getattr(self.cfg, "max_parameters", None)
        if not cap:
            return
        total = sum(p.numel() for p in self.parameters())
        if total > cap:
            c = self.param_counts()
            raise ValueError(
                f"model has {total:,} parameters, over the {cap:,} cap.\n"
                f"  backbone   {c['backbone']:>12,}\n"
                f"  projection {c['projection']:>12,}\n"
                f"  AMSM       {c['amsm']:>12,}\n"
                f"  DAM        {c['dam']:>12,}\n"
                f"  head       {c['head']:>12,}\n"
                f"Options: pick a smaller backbone (python backbones.py lists "
                f"them), lower proj_channels (AMSM cost grows as C^2), or "
                f"raise/disable the cap with ModelConfig.max_parameters."
            )

    # -- forward ------------------------------------------------------------
    def forward(self, x: torch.Tensor, keep_maps: bool = False) -> Dict[str, torch.Tensor]:
        return self.forward_from_feats(self.backbone(x), keep_maps)

    def forward_from_feats(self, feats: List[torch.Tensor],
                           keep_maps: bool = False) -> Dict[str, torch.Tensor]:
        """Everything after the backbone. Lets distillation reuse the backbone maps."""
        gates: List[torch.Tensor] = []
        pooled: List[torch.Tensor] = []
        maps: List[torch.Tensor] = []

        for i, f in enumerate(feats):
            if self.proj is not None:
                f = self.proj[i](f)
            if self.amsm is not None:
                f, w = self.amsm[i](f)
                gates.append(w)
            if self.dam is not None:
                f = self.dam[i](f)
            if keep_maps:
                f.retain_grad()
                maps.append(f)
            pooled.append(f.mean(dim=(2, 3)))

        self._last_maps = maps
        z = torch.cat(pooled, dim=1)
        logits = self.head(z)

        out: Dict[str, torch.Tensor] = {"logits": logits, "features": z}
        if gates:
            out["gates"] = torch.stack(gates, dim=1)   # [B, n_stages, 4]

        if self.cfg.head_type == "edl":
            out.update(evidential_outputs(logits, self.cfg.num_classes))
        else:
            # 'ce' and 'cefocal' share the same softmax head. Their scalar
            # uncertainty is 1 - max p, so the uncertainty analyses in
            # evaluate.py run identically across all three losses and the
            # evidential vacuity has something to be compared against.
            probs = torch.softmax(logits, dim=1)
            out["probs"] = probs
            out["uncertainty"] = 1.0 - probs.max(dim=1).values
        return out

    # -- convenience --------------------------------------------------------
    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """P(defective). The positive class is index 1."""
        return self.forward(x)["probs"][:, 1]

    def enable_mc_dropout(self) -> None:
        """Dropout layers only into train mode, for the MC-Dropout baseline."""
        self.eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    def param_counts(self) -> Dict[str, int]:
        def n(mod: Optional[nn.Module]) -> int:
            return 0 if mod is None else sum(p.numel() for p in mod.parameters())

        return {
            "backbone": n(self.backbone),
            "projection": n(self.proj),
            "amsm": n(self.amsm),
            "dam": n(self.dam),
            "head": n(self.head),
            "total": sum(p.numel() for p in self.parameters()),
            "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }

    def describe(self) -> str:
        c = self.param_counts()
        cap = getattr(self.cfg, "max_parameters", None)
        budget = (f"\nbudget          {c['total']:,} / {cap:,} "
                  f"({100 * c['total'] / cap:.0f}% of cap)" if cap else "")
        return (
            f"backbone        {self.cfg.backbone} ({self.backbone.spec.norm_style}-norm)\n"
            f"strides         {self.backbone.strides}\n"
            f"stage channels  {self.backbone.out_channels} -> {self.stage_channels}\n"
            f"AMSM            {'on' if self.amsm else 'off'} "
            f"(dilations {self.cfg.amsm_dilations}, norm {self.amsm_norm})\n"
            f"DAM             {'on' if self.dam else 'off'}\n"
            f"head            {self.cfg.head_type} "
            f"({self.feat_dim} -> {' -> '.join(map(str, self.cfg.head_hidden))} "
            f"-> {self.cfg.num_classes})\n"
            f"parameters      total {c['total']:,}  "
            f"(backbone {c['backbone']:,}, proj {c['projection']:,}, "
            f"AMSM {c['amsm']:,}, head {c['head']:,})"
            + budget
        )


# Backwards-compatible alias.
HSANetPV = TabletDefectNet


# ---------------------------------------------------------------------------
# Ablation variants
# ---------------------------------------------------------------------------
VARIANTS = {
    "backbone": dict(use_amsm=False, use_dam=False),
    "dam_only": dict(use_amsm=False, use_dam=True),
    "amsm_only": dict(use_amsm=True, use_dam=False),   # proposed
    "amsm_dam": dict(use_amsm=True, use_dam=True),
}


def build_model(
    variant: str = "amsm_only",
    head_type: str = "edl",
    backbone: str = MODEL.backbone,
    strides: Sequence[int] = MODEL.feature_strides,
    proj_channels: Optional[int] = MODEL.proj_channels,
    amsm_norm: Optional[str] = None,
    num_classes: int = 2,
    pretrained: bool = True,
    max_parameters: Optional[int] = MODEL.max_parameters,
) -> TabletDefectNet:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant '{variant}'; choose from {list(VARIANTS)}")
    if head_type not in ("edl", "ce", "cefocal"):
        raise ValueError("head_type must be 'edl', 'ce' or 'cefocal'")
    if backbone not in REGISTRY:
        raise ValueError(f"unknown backbone '{backbone}'. Available:\n{describe_backbones()}")

    cfg = ModelConfig(
        **{**VARIANTS[variant],
           "backbone": backbone,
           "feature_strides": tuple(strides),
           "proj_channels": proj_channels,
           "amsm_norm": amsm_norm,
           "head_type": head_type,
           "num_classes": num_classes,
           "pretrained": pretrained,
           "max_parameters": max_parameters}
    )
    return TabletDefectNet(cfg)


if __name__ == "__main__":
    m = build_model(pretrained=False)
    print(m.describe())
    out = m(torch.randn(2, 3, 224, 224))
    print("\noutputs:", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)})
    print(f"vacuity u range: [{out['uncertainty'].min():.3f}, "
          f"{out['uncertainty'].max():.3f}]  (bounded in (0, 1])")
