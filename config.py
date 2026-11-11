"""
Central configuration.

Everything under PV_PROTOCOL is fixed by the PharmaVision benchmark paper
(Mahmood et al., Pharmaceutics 2026, 18, 1287) and MUST NOT be changed if you
want your numbers to sit in the same table as their twelve baselines.

Everything under MODEL / LOSS is the proposed method (HSANet idea, adapted).
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------------
# PV protocol -- frozen. Section references are to Mahmood et al. 2026.
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class PVProtocol:
    # Sect. 2.1.5 -- fixed splits, class balanced
    n_train_per_class: int = 190
    n_val_per_class: int = 30
    n_test_per_class: int = 100

    # Sect. 2.1 -- "the defective class was treated as the positive class"
    positive_class: str = "defective"
    class_names: Tuple[str, str] = ("normal", "defective")

    # Sect. 2.3 -- preprocessing identical for every architecture
    image_size: int = 224
    imagenet_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    imagenet_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # Sect. 2.1.6 -- offline augmentation, 2 variants per training image
    aug_variants_per_image: int = 2
    aug_rotation_deg: float = 12.0
    aug_brightness_contrast_jitter: float = 0.07
    # 380 originals -> 1140 total
    expected_train_images_after_aug: int = 1140

    # Sect. 2.3 -- optimizer and schedule
    optimizer: str = "adam"           # Adam, NOT AdamW
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 16
    max_epochs: int = 100
    warmup_epochs: int = 1            # linear warm-up over the first epoch
    cosine_final_lr_frac: float = 0.01  # cosine decay to 1% of base lr

    # Sect. 2.3 -- checkpoint selection and early stopping
    selection_metric: str = "f2"      # val F2-score, not accuracy and not F1
    decision_threshold: float = 0.40  # fixed before any test evaluation
    early_stopping_patience: int = 15

    # Sect. 2.3 -- single run, fixed seed
    seed: int = 7

    # Sect. 3.3 -- statistics
    bootstrap_resamples: int = 10_000
    confidence_level: float = 0.95

    # Sect. 2.4 -- efficiency profiling
    latency_warmup_runs: int = 50
    latency_timed_runs: int = 500

    # Sect. 2.1.7 -- cross-dataset evaluation
    pillqc_classes: Tuple[str, str] = ("normal", "chip")  # 'dirt' excluded
    pillqc_expected_counts: Tuple[int, int] = (149, 43)


PV = PVProtocol()


# ----------------------------------------------------------------------------
# Proposed model (HSANet adapted to binary tablet defect classification)
# ----------------------------------------------------------------------------
@dataclass
class ModelConfig:
    # ImageNet-pretrained backbone. See backbones.py for the registry. ConvNeXt-
    # Tiny is the default: LayerNorm throughout (so AMSM uses GroupNorm and no
    # batch statistics are involved), 27.8M parameters, and the ConvNeXt family
    # is the experimental focus. The 5M cap that once selected EfficientNet-B0
    # has been lifted (see max_parameters below).
    backbone: str = "convnext_tiny"
    # Feature taps by STRIDE, not by block index, and discovered at runtime, so
    # backbones are swappable. Add 4 for a stride-4 tap if hairline cracks are
    # being missed -- but check the parameter budget afterwards.
    feature_strides: Tuple[int, ...] = (8, 16, 32)
    pretrained: bool = True

    # 1x1 projection before AMSM. None (default) feeds AMSM the full backbone
    # width, so no feature channels are discarded. Set an integer to cap the
    # width and reduce AMSM cost, which grows as C^2.
    proj_channels: Optional[int] = None

    # Parameter ceiling, checked at construction. Disabled (None) because the
    # project no longer constrains model size. For reference, the PV baselines
    # range from 1.26M (ShuffleNetV2) to 139.58M (VGG-19). Pass
    # --max-params 5000000 to train.py to re-enable the check for compact runs.
    max_parameters: Optional[int] = None

    use_amsm: bool = True
    use_dam: bool = False   # ablated in the source paper and excluded; see README
    amsm_dilations: Tuple[int, int, int] = (1, 2, 4)
    # None = follow the backbone ('gn' for LayerNorm backbones, 'bn' otherwise).
    amsm_norm: Optional[str] = None
    # NOTE: the source paper's text says hidden = max(C/4, 16) but its own
    # parameter table is only reproducible with hidden = C // 4. We use C // 4.
    amsm_gate_hidden_div: int = 4

    num_classes: int = 2
    head_hidden: Tuple[int, int] = (256, 128)
    head_dropout: Tuple[float, float] = (0.3, 0.2)

    # 'edl' = full proposed system; 'ce' = same architecture, plain CE.
    # Both are reported -- see README "loss fairness".
    head_type: str = "edl"


@dataclass
class LossConfig:
    lambda_kl: float = 0.2            # annealed min(1, t/10) * lambda_kl
    kl_anneal_epochs: int = 10
    lambda_focal: float = 0.3
    focal_gamma: float = 2.0
    # PV splits are 1:1 balanced (190/190), so no class weighting is applied.
    # This deliberately departs from HSANet's inverse-frequency weighting,
    # which was motivated by BRISC class imbalance that PV does not have.
    class_weights: List[float] = field(default_factory=lambda: [1.0, 1.0])


@dataclass
class PathConfig:
    pv_root: str = "data/PV"
    pv_augmented_root: str = "data/PV_augmented"
    pillqc_root: str = "data/pillQC"
    output_dir: str = "runs"
    cache_dir: str = ".cache"


MODEL = ModelConfig()
LOSS = LossConfig()
PATHS = PathConfig()


def run_dir_name(
    backbone: str = MODEL.backbone,
    variant: str = "amsm_only",
    head: str = MODEL.head_type,
    seed: int = PV.seed,
) -> str:
    """Canonical run-directory name.

    Single source of truth, used by train.py when writing and by smoke_test.py
    when reading. Defined here because having each script build the name
    itself let them drift: smoke_test once looked for a 'convnext_tiny_...'
    directory after the default backbone had changed to EfficientNet-B0, and
    the mismatch only surfaced after four training runs had completed.
    """
    return f"{backbone}_{variant}_{head}_seed{seed}"
