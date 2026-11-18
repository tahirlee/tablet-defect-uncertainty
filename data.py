"""
PharmaVision (PV) and pillQC data loading.

PV layout (after running augment.py on the training images):

    data/PV_augmented/
      train/{normal,defective}/   570 + 570 images (190 originals + 2 augmented copies each)
      val/{normal,defective}/      30 +  30
      test/{normal,defective}/    100 + 100

pillQC: the normal (149) and chip (43) images are used; dirt is excluded.
Every image is resized to 224 x 224 and normalized with ImageNet statistics.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
SPLITS = {"train": {"train", "training"}, "val": {"val", "valid", "validation"},
          "test": {"test", "testing"}}
CLASSES = {"normal": 0, "good": 0, "defective": 1, "defect": 1, "bad": 1}
EXPECTED = {"train": (570, 570), "val": (30, 30), "test": (100, 100)}   # (normal, defective)


def transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=list(MEAN), std=list(STD)),
    ])


def _norm(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def list_pv(root: str | Path) -> Dict[str, List[Tuple[Path, int]]]:
    """{split: [(image path, label), ...]}, label 1 = defective, sorted by path."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"PV dataset not found at '{root}'.")
    out: Dict[str, List[Tuple[Path, int]]] = {"train": [], "val": [], "test": []}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXT:
            continue
        split = label = None
        for part in map(_norm, path.relative_to(root).parts[:-1]):
            split = split or next((s for s, names in SPLITS.items() if part in names), None)
            label = label if label is not None else CLASSES.get(part)
        if split is not None and label is not None:
            out[split].append((path, label))
    for split in out:
        out[split].sort(key=lambda t: str(t[0]))
        n0 = sum(1 for _, y in out[split] if y == 0)
        n1 = len(out[split]) - n0
        if (n0, n1) != EXPECTED[split]:
            raise RuntimeError(f"PV {split}: expected {EXPECTED[split]} (normal, defective) "
                               f"images, found {(n0, n1)}. Check the folder layout.")
    return out


def list_pillqc(root: str | Path) -> List[Tuple[Path, int]]:
    """pillQC normal (0) and chip (1) images; dirt is excluded.

    The chip folder may also be named "defective".
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"pillQC not found at '{root}'.")
    items = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMG_EXT:
            continue
        names = set(map(_norm, path.relative_to(root).parts[:-1]))
        if names & {"dirt", "dirty"}:
            continue
        if names & {"chip", "chips", "chipped", "defect", "defective"}:
            items.append((path, 1))
        elif names & {"normal", "good"}:
            items.append((path, 0))
    n0 = sum(1 for _, y in items if y == 0)
    if (n0, len(items) - n0) != (149, 43):
        raise RuntimeError(f"pillQC: expected 149 normal and 43 chip images, "
                           f"found {n0} and {len(items) - n0}.")
    return items


class TabletDataset(Dataset):
    def __init__(self, items, tf=None):
        self.items = list(items)
        self.tf = tf or transform()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        path, label = self.items[i]
        return self.tf(Image.open(path).convert("RGB")), label, str(path)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def make_loader(items, batch_size: int = 16, shuffle: bool = False, seed: int = 0,
                num_workers: int = 4) -> DataLoader:
    g = None
    if shuffle:
        g = torch.Generator()
        g.manual_seed(seed)
    return DataLoader(TabletDataset(items), batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(),
                      generator=g)
