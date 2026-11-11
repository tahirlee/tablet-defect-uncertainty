"""
PharmaVision (PV) and pillQC data pipeline.

Implements the PV protocol exactly:
  - fixed tablet-level splits (190/190 train, 30/30 val, 100/100 test)
  - offline augmentation of the TRAINING partition only, 380 -> 1140 images
  - 224x224 resize + ImageNet normalization for every split
  - no augmentation at validation or test time beyond resize + normalize
  - pillQC restricted to the normal and chip classes (dirt excluded)
"""
from __future__ import annotations

import json
import os
import random
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import PV, PATHS

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Directory-name aliases, so the loader survives however the release is laid out.
NORMAL_ALIASES = {"normal", "good", "ok", "non_defective", "nondefective", "0"}
DEFECT_ALIASES = {"defective", "defect", "bad", "ng", "abnormal", "1"}
SPLIT_ALIASES = {
    "train": {"train", "training", "train_set"},
    "val": {"val", "valid", "validation", "val_set"},
    "test": {"test", "testing", "test_set"},
}


# ---------------------------------------------------------------------------
# Transforms (PV Sect. 2.3)
# ---------------------------------------------------------------------------
def build_transform() -> transforms.Compose:
    """Single preprocessing pipeline shared by every split and every model.

    PV applies augmentation OFFLINE to the training partition only, so the
    runtime transform is identical for train/val/test. This is intentional.
    """
    return transforms.Compose([
        transforms.Resize((PV.image_size, PV.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(PV.imagenet_mean), std=list(PV.imagenet_std)),
    ])


# ---------------------------------------------------------------------------
# Filesystem discovery
# ---------------------------------------------------------------------------
def _is_image(p: Path) -> bool:
    return p.suffix.lower() in IMG_EXT


def _label_of_dir(name: str) -> Optional[int]:
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    if n in NORMAL_ALIASES:
        return 0
    if n in DEFECT_ALIASES:
        return 1
    return None


def _split_of_dir(name: str) -> Optional[str]:
    n = name.strip().lower().replace("-", "_").replace(" ", "_")
    for split, aliases in SPLIT_ALIASES.items():
        if n in aliases:
            return split
    return None


def discover_pv(root: str | Path) -> Dict[str, List[Tuple[Path, int]]]:
    """Walk the PV release and return {split: [(path, label), ...]}.

    Accepts either {split}/{class}/*.jpg or {class}/{split}/*.jpg.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"PV dataset not found at '{root}'. Download it from "
            "https://github.com/tahirlee/pharmaceutical-tablet-defect-detection "
            "(or Zenodo 10.5281/zenodo.22773411) and point PATHS.pv_root at it."
        )

    out: Dict[str, List[Tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_image(path):
            continue
        parts = [p for p in path.relative_to(root).parts[:-1]]
        split = label = None
        for part in parts:
            if split is None:
                split = _split_of_dir(part)
            if label is None:
                label = _label_of_dir(part)
        if split is None or label is None:
            continue
        out[split].append((path, label))

    for split in out:
        out[split].sort(key=lambda t: str(t[0]))
    return out


def verify_pv_counts(index: Dict[str, List[Tuple[Path, int]]], augmented: bool = False) -> None:
    """Fail loudly if the split sizes do not match the published protocol."""
    expect = {
        "train": (PV.n_train_per_class, PV.n_train_per_class),
        "val": (PV.n_val_per_class, PV.n_val_per_class),
        "test": (PV.n_test_per_class, PV.n_test_per_class),
    }
    if augmented:
        k = 1 + PV.aug_variants_per_image
        expect["train"] = (PV.n_train_per_class * k, PV.n_train_per_class * k)

    problems = []
    for split, (exp_n, exp_d) in expect.items():
        items = index.get(split, [])
        got_n = sum(1 for _, y in items if y == 0)
        got_d = sum(1 for _, y in items if y == 1)
        if (got_n, got_d) != (exp_n, exp_d):
            problems.append(
                f"  {split}: expected {exp_n} normal / {exp_d} defective, "
                f"found {got_n} / {got_d}"
            )
    if problems:
        raise RuntimeError(
            "PV split sizes do not match the published protocol:\n"
            + "\n".join(problems)
            + "\n\nYour results would not be comparable to the paper's twelve "
              "baselines. Check the dataset layout before training."
        )


PILLQC_CHIP_ALIASES = {"chip", "chips", "chipped", "defect", "defective"}
PILLQC_NORMAL_ALIASES = {"normal", "good", "ok", "non_defective", "nondefective"}
PILLQC_DIRT_ALIASES = {"dirt", "dirty", "dirtied"}


def discover_pillqc(root: str | Path) -> List[Tuple[Path, int]]:
    """pillQC, restricted to normal (0) and chip (1). Dirt is excluded.

    PV excludes dirt on conceptual grounds: it is deposited foreign particulate,
    not an intrinsic structural defect, so including it would change the meaning
    of the positive class mid-evaluation (PV Sect. 2.1.7).

    Folder names are matched case-insensitively with common aliases. The counts
    must equal the PV paper's (149 normal, 43 chip); otherwise this raises, so a
    broken folder layout cannot produce cross-dataset numbers.
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(
            f"pillQC not found at '{root}'. Clone "
            "https://github.com/matlab-deep-learning/pillQC"
        )
    items: List[Tuple[Path, int]] = []
    skipped_dirs = set()
    top_level = set()
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not _is_image(path):
            continue
        rel = path.relative_to(root)
        top_level.add(rel.parts[0] if len(rel.parts) > 1 else "(files at top level)")
        names = {p.strip().lower().replace("-", "_").replace(" ", "_")
                 for p in rel.parts[:-1]}
        if names & PILLQC_DIRT_ALIASES:
            continue
        if names & PILLQC_CHIP_ALIASES:
            items.append((path, 1))
        elif names & PILLQC_NORMAL_ALIASES:
            items.append((path, 0))
        else:
            skipped_dirs.add(str(rel.parent))

    n_norm = sum(1 for _, y in items if y == 0)
    n_chip = sum(1 for _, y in items if y == 1)
    exp_norm, exp_chip = PV.pillqc_expected_counts
    if (n_norm, n_chip) != (exp_norm, exp_chip):
        raise RuntimeError(
            f"pillQC has {n_norm} normal and {n_chip} chip images; the PV paper "
            f"uses {exp_norm} normal and {exp_chip} chip.\n"
            f"Top-level folders found: {sorted(top_level)}\n"
            f"Folders not recognised (skipped): {sorted(skipped_dirs)[:10]}\n"
            "Rename the folders to normal/ and chip/, or add the name to "
            "PILLQC_CHIP_ALIASES / PILLQC_NORMAL_ALIASES in data.py."
        )
    return items


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TabletDataset(Dataset):
    """Images and binary labels. Nothing else is used."""

    def __init__(self, items: Sequence[Tuple[Path, int]], transform=None):
        self.items = list(items)
        self.transform = transform or build_transform()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label = self.items[idx]
        img = Image.open(path).convert("RGB")
        x = self.transform(img)
        return x, label, str(path)


# ---------------------------------------------------------------------------
# Offline augmentation (PV Sect. 2.1.6)
# ---------------------------------------------------------------------------
def _augment_once(img: Image.Image, rng: random.Random) -> Image.Image:
    """Mild geometric + illumination transform.

    PV deliberately EXCLUDES random cropping, Gaussian blur/noise, cutout and
    hue/saturation shifting, because tablet defects are small in extent or
    colour-dependent and those transforms destroy the evidence the classifier
    is meant to detect. Do not add them.
    """
    from PIL import ImageEnhance

    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)

    angle = rng.uniform(-PV.aug_rotation_deg, PV.aug_rotation_deg)
    # Fill exposed corners by sampling the source image, per PV Sect. 2.1.6.
    fill = tuple(int(c) for c in np.asarray(img).reshape(-1, 3).mean(axis=0))
    img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=fill)

    j = PV.aug_brightness_contrast_jitter
    img = ImageEnhance.Brightness(img).enhance(1.0 + rng.uniform(-j, j))
    img = ImageEnhance.Contrast(img).enhance(1.0 + rng.uniform(-j, j))
    return img


def build_augmented_training_set(
    pv_root: str | Path = PATHS.pv_root,
    out_root: str | Path = PATHS.pv_augmented_root,
    seed: int = PV.seed,
) -> Path:
    """Generate and SAVE augmented training images (offline, as PV specifies).

    Produces 190*3 = 570 images per class, 1140 total, preserving 1:1 balance.
    Validation and test partitions are copied through untouched.
    """
    out_root = Path(out_root)
    index = discover_pv(pv_root)
    verify_pv_counts(index, augmented=False)

    rng = random.Random(seed)
    cls_dir = {0: "normal", 1: "defective"}

    # Originals are COPIED byte-for-byte, never re-encoded. Re-encoding a JPEG
    # at PIL's default quality (75) can erase the fine edges of hairline cracks
    # and would leave the PV test set preprocessed differently from pillQC,
    # which is read directly from disk.
    for split in ("val", "test"):
        for path, label in index[split]:
            dst = out_root / split / cls_dir[label] / path.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(path, dst)

    for path, label in index["train"]:
        dst_dir = out_root / "train" / cls_dir[label]
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dst_dir / path.name)
        img = Image.open(path).convert("RGB")
        is_jpeg = path.suffix.lower() in {".jpg", ".jpeg"}
        for k in range(PV.aug_variants_per_image):
            aug = _augment_once(img, rng)
            out = dst_dir / f"{path.stem}_aug{k + 1}{path.suffix}"
            if is_jpeg:
                aug.save(out, quality=95)   # high quality: keep defect detail
            else:
                aug.save(out)

    out_index = discover_pv(out_root)
    verify_pv_counts(out_index, augmented=True)
    total = len(out_index["train"])
    print(f"[ok] augmented training set written to {out_root} ({total} images)")
    if total != PV.expected_train_images_after_aug:
        print(f"[warn] expected {PV.expected_train_images_after_aug}, got {total}")
    return out_root


# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
def seed_everything(seed: int = PV.seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    # PV Sect. 2.3 notes that non-deterministic CUDA kernels were NOT disabled,
    # so bit-exact reproduction requires the same hardware/software stack.


def make_loaders(
    train_root: str | Path,
    batch_size: int = PV.batch_size,
    num_workers: int = 4,
    seed: int = PV.seed,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    index = discover_pv(train_root)
    verify_pv_counts(index, augmented=(len(index["train"]) > 2 * PV.n_train_per_class))
    tf = build_transform()

    g = torch.Generator()
    g.manual_seed(seed)

    def _loader(split: str, shuffle: bool) -> DataLoader:
        ds = TabletDataset(index[split], tf)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            generator=g if shuffle else None,
        )

    return _loader("train", True), _loader("val", False), _loader("test", False)


def make_pillqc_loader(
    root: str | Path = PATHS.pillqc_root,
    batch_size: int = PV.batch_size,
    num_workers: int = 4,
) -> DataLoader:
    items = discover_pillqc(root)
    ds = TabletDataset(items, build_transform())
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Prepare the PV augmented training set.")
    ap.add_argument("--pv-root", default=PATHS.pv_root)
    ap.add_argument("--out-root", default=PATHS.pv_augmented_root)
    args = ap.parse_args()
    build_augmented_training_set(args.pv_root, args.out_root)
