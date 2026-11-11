"""
Pluggable ImageNet-pretrained backbones, with a parameter budget in mind.

Feature taps are DISCOVERED AT RUNTIME, not hardcoded: the backbone is probed
once with a dummy 224x224 input, every block's output shape is recorded, and
the deepest block producing each requested stride is selected. This is robust
across torchvision versions and removes a whole class of silent error (a wrong
hardcoded channel count would only surface as a shape mismatch much later).

Blocks after the last tap are DROPPED. This matters for the parameter budget:
EfficientNet-B0's final 1x1 convolution to 1280 channels costs ~412k
parameters and is never used when the deepest tap is stride 32 at 320
channels.

Default: convnext_tiny (the 5M cap was lifted; see config.py).

  - At 224x224 it taps 40 / 112 / 320 channels at strides 8 / 16 / 32, a stage
    structure close to the one AMSM was designed against (EfficientNet-B3's
    48 / 136 / 384), so the module's dilated branches operate on the kind of
    feature widths they were tuned for.
  - It is NOT one of PV's twelve baselines. EfficientNetV2-S, ResNet-50,
    Xception, ShuffleNetV2 and MobileNetV3-small are -- avoid those as the
    proposed encoder unless you intend AMSM + EDL to carry the whole claim.
  - PV's two compact baselines transferred worst of all the CNNs
    (MobileNetV3-small -26.45 pp F1 to pillQC, ShuffleNetV2 -35.34 pp), while
    GoogLeNet at 5.60M lost only 6.87 pp. A sub-5M model with GoogLeNet-class
    transfer is therefore a well-defined and genuinely open target.

HONEST CAVEAT on the batch-size argument. ConvNeXt was previously the default
partly because LayerNorm is batch-size independent, which suits the protocol's
fixed batch size of 16. No sub-5M torchvision backbone is LayerNorm-based, so
that specific advantage is given up here. Two things soften it: the AMSM and
projection blocks still use GroupNorm (`amsm_norm='gn'`), and every one of
PV's twelve baselines was itself a BatchNorm network trained at batch size 16,
so this is the benchmark's shared condition rather than a handicap unique to
this model. If you want the LayerNorm property back inside the budget, install
timm and use `convnext_atto` or `convnext_femto` (registered below).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn


@dataclass
class BackboneSpec:
    name: str
    builder: Callable          # (pretrained: bool) -> nn.Sequential of blocks
    norm_style: str            # 'ln' or 'bn' -- decides the AMSM default norm
    source: str                # 'torchvision' or 'timm'
    note: str = ""
    is_pv_baseline: bool = False
    # Trailing blocks to exclude from tap selection. The EfficientNet and
    # MobileNet families end with a 1x1 convolution that widens the last stage
    # enormously (B0 320->1280, B3 384->1536, V2-S 256->1280, MNv2 320->1280,
    # MNv3-L 160->960) purely to feed a global-pool + linear classifier. It
    # sits at the same stride as the real last stage, so "deepest block at
    # stride 32" would select it: the tap would be 1280 channels instead of
    # 320, the projection would become a 10x bottleneck, and efficientnet_b3
    # would stop reproducing the source paper's 48/136/384 structure.
    # Set explicitly per backbone so this is deterministic, not heuristic.
    # None = infer (see _infer_strip_trailing); 0 = strip nothing.
    strip_trailing: Optional[int] = 0
    # Stage channels expected at strides 8/16/32 for a 224px input. Checked by
    # the self-test, so a torchvision version change cannot silently alter the
    # architecture.
    expected_channels: Optional[Tuple[int, ...]] = None


# ---------------------------------------------------------------------------
# Builders -- each returns a flat nn.Sequential that can be probed block by
# block. Classifier heads are discarded.
# ---------------------------------------------------------------------------
def _tv(fn_name: str, weights_name: str):
    def build(pretrained: bool):
        import torchvision.models as M

        fn = getattr(M, fn_name)
        w = getattr(M, weights_name).DEFAULT if pretrained else None
        return fn(weights=w).features

    return build


def _tv_resnet(fn_name: str, weights_name: str):
    def build(pretrained: bool):
        import torchvision.models as M

        fn = getattr(M, fn_name)
        w = getattr(M, weights_name).DEFAULT if pretrained else None
        net = fn(weights=w)
        return nn.Sequential(
            nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool),
            net.layer1, net.layer2, net.layer3, net.layer4,
        )

    return build


def _tv_regnet(fn_name: str, weights_name: str):
    def build(pretrained: bool):
        import torchvision.models as M

        fn = getattr(M, fn_name)
        w = getattr(M, weights_name).DEFAULT if pretrained else None
        net = fn(weights=w)
        return nn.Sequential(net.stem, *list(net.trunk_output.children()))

    return build


def _tv_shufflenet(fn_name: str, weights_name: str):
    """ShuffleNetV2 has no `.features`; its stages are attributes, so list them."""
    def build(pretrained: bool):
        import torchvision.models as M

        fn = getattr(M, fn_name)
        w = getattr(M, weights_name).DEFAULT if pretrained else None
        net = fn(weights=w)
        return nn.Sequential(net.conv1, net.maxpool, net.stage2, net.stage3,
                             net.stage4, net.conv5)

    return build


class _InvRes(nn.Module):
    """MobileNetV2-style inverted residual: 1x1 expand, depthwise 3x3, 1x1 project."""

    def __init__(self, cin: int, cout: int, stride: int, expand: int = 2):
        super().__init__()
        hidden = cin * expand
        self.use_res = stride == 1 and cin == cout
        self.body = nn.Sequential(
            nn.Conv2d(cin, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, hidden, 3, stride, 1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU6(inplace=True),
            nn.Conv2d(hidden, cout, 1, bias=False), nn.BatchNorm2d(cout),
        )

    def forward(self, x):
        y = self.body(x)
        return x + y if self.use_res else y


def _edgepill(width: float):
    """EdgePillNet encoder: stride-2 stages with taps at strides 8/16/32.

    Trained from scratch. There are no ImageNet weights for this network, so the
    `pretrained` argument is ignored.
    """
    def ch(c: int) -> int:
        return max(8, int(round(c * width)))

    def build(pretrained: bool):
        return nn.Sequential(
            nn.Sequential(nn.Conv2d(3, ch(16), 3, 2, 1, bias=False),
                          nn.BatchNorm2d(ch(16)), nn.ReLU6(inplace=True)),  # stride 2
            _InvRes(ch(16), ch(24), 2),    # stride 4
            _InvRes(ch(24), ch(32), 2),    # stride 8   -> tap (deepest at 8)
            _InvRes(ch(32), ch(64), 2),    # stride 16
            _InvRes(ch(64), ch(64), 1),    # stride 16  -> tap (deepest at 16)
            _InvRes(ch(64), ch(128), 2),   # stride 32
            _InvRes(ch(128), ch(128), 1),  # stride 32  -> tap (deepest at 32)
        )

    return build


def _timm(model_name: str):
    def build(pretrained: bool):
        import timm

        net = timm.create_model(model_name, pretrained=pretrained,
                                features_only=True)
        # features_only models expose .feature_info; wrap so probing still works
        return _TimmWrapper(net)

    return build


class _TimmWrapper(nn.Module):
    """Adapts a timm features_only model to the block-sequence interface."""

    def __init__(self, net):
        super().__init__()
        self.net = net

    def __len__(self):
        return len(self.net.feature_info.channels())

    def __iter__(self):
        # Probing iterates blocks; timm gives all stages in one call instead,
        # so expose pseudo-blocks that replay the cached forward.
        raise TypeError("timm backbones are probed via forward_all")

    def forward_all(self, x) -> List[torch.Tensor]:
        return list(self.net(x))


# ---------------------------------------------------------------------------
REGISTRY: Dict[str, BackboneSpec] = {
    # ---- compact (fit a <5M total budget) ---------------------------------
    "efficientnet_b0": BackboneSpec(
        "efficientnet_b0", _tv("efficientnet_b0", "EfficientNet_B0_Weights"),
        "bn", "torchvision",
        "DEFAULT. ~3.6M once the 1280-channel classifier conv is stripped. "
        "Stage widths closest to the EfficientNet-B3 structure AMSM was "
        "designed against.",
        strip_trailing=1, expected_channels=(40, 112, 320),
    ),
    "mobilenet_v3_large": BackboneSpec(
        "mobilenet_v3_large", _tv("mobilenet_v3_large", "MobileNetV3_Large_Weights"),
        "bn", "torchvision",
        "~2.8M. Smallest sensible option; squeeze-excite + hard-swish. "
        "Note MobileNetV3-SMALL is a PV baseline, Large is not.",
        strip_trailing=1, expected_channels=(40, 112, 160),
    ),
    "mobilenet_v2": BackboneSpec(
        "mobilenet_v2", _tv("mobilenet_v2", "MobileNet_V2_Weights"),
        "bn", "torchvision",
        "~1.8M once stripped. Plainest compact baseline, no attention.",
        strip_trailing=1, expected_channels=(32, 96, 320),
    ),
    # Student backbone for teacher-student distillation (distill_student.py).
    # The 192 -> 1024 conv5 at stride 32 is stripped; the deepest kept tap is
    # stage4 (192 channels at stride 32).
    "shufflenet_v2_x0_5": BackboneSpec(
        "shufflenet_v2_x0_5",
        _tv_shufflenet("shufflenet_v2_x0_5", "ShuffleNet_V2_X0_5_Weights"),
        "bn", "torchvision",
        "Student encoder, ImageNet-pretrained, classifier and conv5 removed. "
        "Not checked against PV's ShuffleNetV2 configuration (1.26M); verify "
        "the parameter count with param_budget.py.",
        strip_trailing=1, expected_channels=(48, 96, 192),
    ),
    # EdgePillNet encoders: custom, trained from scratch, no ImageNet weights.
    "edgepillnet_w1": BackboneSpec(
        "edgepillnet_w1", _edgepill(1.0), "bn", "custom",
        "Student encoder, width 1.0, from scratch. Taps 32/64/128 channels.",
        strip_trailing=0, expected_channels=(32, 64, 128),
    ),
    "edgepillnet_w15": BackboneSpec(
        "edgepillnet_w15", _edgepill(1.5), "bn", "custom",
        "Student encoder, width 1.5, from scratch. Taps 48/96/192 channels.",
        strip_trailing=0, expected_channels=(48, 96, 192),
    ),
    "regnet_y_400mf": BackboneSpec(
        "regnet_y_400mf", _tv_regnet("regnet_y_400mf", "RegNet_Y_400MF_Weights"),
        "bn", "torchvision",
        "~3.9M. Designed by population search; strong accuracy per FLOP. "
        "No classifier expansion conv, so nothing is stripped.",
        strip_trailing=0, expected_channels=(104, 208, 440),
    ),
    "convnext_atto": BackboneSpec(
        "convnext_atto", _timm("convnext_atto"), "ln", "timm",
        "~3.4M. Keeps the LayerNorm / batch-size-independence property "
        "inside the budget. Requires `pip install timm`.",
        strip_trailing=0, expected_channels=(80, 160, 320),
    ),
    "convnext_femto": BackboneSpec(
        "convnext_femto", _timm("convnext_femto"), "ln", "timm",
        "~4.9M. LayerNorm, slightly larger than atto. Requires timm. "
        "Over the 5M cap once AMSM and the head are added.",
        strip_trailing=0, expected_channels=(96, 192, 384),
    ),
    # ---- larger (over a 5M budget; kept for ablation and reproduction) -----
    "convnext_tiny": BackboneSpec(
        "convnext_tiny", _tv("convnext_tiny", "ConvNeXt_Tiny_Weights"),
        "ln", "torchvision",
        "27.8M. Over budget. LayerNorm; useful as a capacity ceiling.",
        strip_trailing=0, expected_channels=(192, 384, 768),
    ),
    "convnext_base": BackboneSpec(
        "convnext_base", _tv("convnext_base", "ConvNeXt_Base_Weights"),
        "ln", "torchvision",
        "~88M. Largest ConvNeXt here; LayerNorm; ImageNet-1k weights. "
        "Only for runs where the parameter cap is lifted.",
        strip_trailing=0, expected_channels=(256, 512, 1024),
    ),
    "convnext_small": BackboneSpec(
        "convnext_small", _tv("convnext_small", "ConvNeXt_Small_Weights"),
        "ln", "torchvision",
        "~50M. ConvNeXt-Small: same stage widths as Tiny, deeper stage 3 "
        "(27 blocks). LayerNorm; ImageNet-1k weights.",
        strip_trailing=0, expected_channels=(192, 384, 768),
    ),
    "convnext_tiny_22k": BackboneSpec(
        # Same architecture as convnext_tiny, pretrained on ImageNet-22k and
        # fine-tuned on ImageNet-1k (timm tag). Often transfers better to
        # small datasets. Requires `pip install timm`.
        "convnext_tiny_22k", _timm("convnext_tiny.fb_in22k_ft_in1k"),
        "ln", "timm",
        "~28M. ConvNeXt-Tiny with ImageNet-22k pretraining (timm weights).",
        strip_trailing=0, expected_channels=(192, 384, 768),
    ),
    "efficientnet_b3": BackboneSpec(
        "efficientnet_b3", _tv("efficientnet_b3", "EfficientNet_B3_Weights"),
        "bn", "torchvision",
        "~10.7M. The source paper's backbone. With the classifier conv "
        "stripped it taps 48/136/384 -- exactly HSANet's Table 3 structure.",
        strip_trailing=1, expected_channels=(48, 136, 384),
    ),
    "efficientnetv2_s": BackboneSpec(
        "efficientnetv2_s", _tv("efficientnet_v2_s", "EfficientNet_V2_S_Weights"),
        "bn", "torchvision",
        "20.2M. Best in-domain F1 among PV's twelve baselines (88.56%).",
        is_pv_baseline=True,
        strip_trailing=1, expected_channels=(64, 160, 256),
    ),
    "resnet50": BackboneSpec(
        "resnet50", _tv_resnet("resnet50", "ResNet50_Weights"),
        "bn", "torchvision", "23.5M. Reference backbone.", is_pv_baseline=True,
        strip_trailing=0, expected_channels=(512, 1024, 2048),
    ),
}


def _infer_strip_trailing(shapes: List[Tuple[int, int]], ratio: float = 3.0) -> int:
    """Fallback for a backbone registered without an explicit strip_trailing.

    Walks back from the last block and strips any that widens the channel
    count by >= `ratio` at the SAME stride as its predecessor, which is the
    signature of a classifier-feeding 1x1 expansion (B0 320->1280 is 4x,
    MNv3-L 160->960 is 6x). The 3x threshold is deliberately above the
    largest legitimate within-stage widening in the registered networks
    (MobileNetV2's final bottleneck is exactly 2x, 160->320, and must be
    kept), so a 2x threshold would wrongly discard a real stage.

    Prefer setting strip_trailing explicitly. This exists so an unregistered
    backbone degrades sensibly rather than silently tapping 1280 channels.
    """
    n = 0
    i = len(shapes) - 1
    while i > 0 and n < 2:
        (ch, st), (prev_ch, prev_st) = shapes[i], shapes[i - 1]
        if st == prev_st and ch >= ratio * prev_ch:
            n += 1
            i -= 1
        else:
            break
    return n


def available_backbones() -> List[str]:
    return list(REGISTRY)


def describe_backbones() -> str:
    lines = []
    for k, s in REGISTRY.items():
        tag = "  [PV BASELINE]" if s.is_pv_baseline else ""
        lines.append(f"  {k:20s} norm={s.norm_style}  src={s.source}{tag}\n"
                     f"{' ' * 24}{s.note}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------------------------
class MultiStageBackbone(nn.Module):
    """Runs a backbone and returns feature maps at the requested strides.

    Taps are probed at construction, and every block after the last tap is
    discarded so unused trailing layers do not count against the budget.
    """

    def __init__(
        self,
        name: str = "convnext_tiny",
        strides: Sequence[int] = (8, 16, 32),
        pretrained: bool = True,
        input_size: int = 224,
    ):
        super().__init__()
        if name not in REGISTRY:
            raise ValueError(
                f"unknown backbone '{name}'. Available:\n{describe_backbones()}"
            )
        self.spec = REGISTRY[name]
        blocks = self.spec.builder(pretrained)
        self.strides = list(strides)

        if isinstance(blocks, _TimmWrapper):
            self._timm = blocks
            self._probe_timm(input_size)
            return
        self._timm = None

        # -- probe: record (channels, stride) after each block ---------------
        probe = torch.zeros(1, 3, input_size, input_size)
        shapes: List[Tuple[int, int]] = []
        with torch.no_grad():
            cursor = probe
            for block in blocks:
                cursor = block(cursor)
                if cursor.dim() != 4:
                    break
                shapes.append((cursor.shape[1], input_size // cursor.shape[2]))

        # -- exclude trailing classifier-expansion blocks -------------------
        strip = self.spec.strip_trailing
        if strip is None:
            strip = _infer_strip_trailing(shapes)
        self.n_stripped = min(strip, max(0, len(shapes) - len(self.strides)))
        usable = shapes[: len(shapes) - self.n_stripped] if self.n_stripped else shapes

        by_stride: Dict[int, Tuple[int, int]] = {}
        for idx, (ch, st) in enumerate(usable):
            by_stride[st] = (idx, ch)   # later block wins -> deepest per stride

        missing = [s for s in self.strides if s not in by_stride]
        if missing:
            raise ValueError(
                f"{name} produces no feature map at stride(s) {missing} for a "
                f"{input_size}px input; available strides: {sorted(by_stride)}"
            )

        self.tap_indices = [by_stride[s][0] for s in self.strides]
        self.out_channels: List[int] = [by_stride[s][1] for s in self.strides]

        # Drop everything after the last tap. This is what keeps the stripped
        # classifier convolution from counting against the parameter budget.
        last = max(self.tap_indices)
        self.features = nn.Sequential(*list(blocks)[: last + 1])
        self.n_dropped_blocks = len(list(blocks)) - (last + 1)

        if self.spec.expected_channels is not None and \
                tuple(self.strides) == (8, 16, 32) and \
                tuple(self.out_channels) != tuple(self.spec.expected_channels):
            raise RuntimeError(
                f"{name}: tapped channels {self.out_channels} but the registry "
                f"expects {list(self.spec.expected_channels)}. Your torchvision "
                f"or timm version may lay this network out differently. Check "
                f"BackboneSpec.strip_trailing before training."
            )

    def _probe_timm(self, input_size: int) -> None:
        with torch.no_grad():
            feats = self._timm.forward_all(torch.zeros(1, 3, input_size, input_size))
        by_stride = {input_size // f.shape[2]: (i, f.shape[1])
                     for i, f in enumerate(feats)}
        missing = [s for s in self.strides if s not in by_stride]
        if missing:
            raise ValueError(
                f"{self.spec.name} has no stride {missing}; "
                f"available: {sorted(by_stride)}"
            )
        self.tap_indices = [by_stride[s][0] for s in self.strides]
        self.out_channels = [by_stride[s][1] for s in self.strides]
        self.n_dropped_blocks = 0
        self.n_stripped = 0

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self._timm is not None:
            feats = self._timm.forward_all(x)
            return [feats[i] for i in self.tap_indices]

        wanted = set(self.tap_indices)
        collected: Dict[int, torch.Tensor] = {}
        cursor = x
        for i, block in enumerate(self.features):
            cursor = block(cursor)
            if i in wanted:
                collected[i] = cursor
        return [collected[i] for i in self.tap_indices]

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    print("Available backbones:\n" + describe_backbones())
    print(f"\n{'backbone':<20}{'params':>12}{'channels @ s8/16/32':>24}"
          f"{'dropped':>9}  check")
    print("-" * 74)

    failures, skipped = [], []
    for nm in available_backbones():
        try:
            bb = MultiStageBackbone(nm, (8, 16, 32), pretrained=False)
            feats = bb(torch.randn(1, 3, 224, 224))
            strides = [224 // f.shape[2] for f in feats]
            exp = REGISTRY[nm].expected_channels

            ok = strides == [8, 16, 32]
            if exp is not None:
                ok &= tuple(bb.out_channels) == tuple(exp)
            if not ok:
                failures.append(nm)

            print(f"{nm:<20}{bb.param_count():>12,}"
                  f"{str(bb.out_channels):>24}{bb.n_dropped_blocks:>9}  "
                  f"{'ok' if ok else f'MISMATCH (expected {list(exp)})'}")
        except ImportError as exc:
            skipped.append(nm)
            print(f"{nm:<20}{'skipped':>12}{'':>24}{'':>9}  "
                  f"(needs {exc.name}: pip install {exc.name})")
        except Exception as exc:
            failures.append(nm)
            print(f"{nm:<20}{'FAILED':>12}{'':>24}{'':>9}  "
                  f"({type(exc).__name__}: {exc})")

    print("-" * 74)
    if skipped:
        print(f"skipped (optional dependency): {', '.join(skipped)}")
    if failures:
        raise SystemExit(
            f"\nFAILED: {', '.join(failures)}. Tapped channels do not match the "
            "registry. Fix BackboneSpec.strip_trailing before training -- a "
            "wrong tap trains fine and silently changes the architecture."
        )
    print("PASS: every available backbone taps the expected stage channels.")
    print("      efficientnet_b3 -> [48, 136, 384] reproduces HSANet Table 3.")
