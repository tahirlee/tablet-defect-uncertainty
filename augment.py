#!/usr/bin/env python3
"""Create defect-safe offline augmentations for a two-class tablet dataset.

Input layout:
  Train/normal/*.jpg
  Train/defective/*.jpg

Output layout (new folder, created next to the input folder):
  train_aug/normal/*
  train_aug/defective/*

Each source image is copied unchanged and then receives N mild variants. This
never augments Val or test: doing so would contaminate evaluation.
"""
import argparse
import random
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_OUTPUT_NAME = 'train_aug'


def resolve_input_dir(raw):
    """Accept Train/train/TRAIN without the caller worrying about case."""
    candidate = Path(raw)
    if candidate.is_dir():
        return candidate
    parent = candidate.parent if str(candidate.parent) else Path('.')
    if parent.is_dir():
        wanted = candidate.name.lower()
        for sibling in sorted(parent.iterdir()):
            if sibling.is_dir() and sibling.name.lower() == wanted:
                return sibling
    raise FileNotFoundError(
        f'Input directory not found: {candidate}. Pass --input_dir with the path '
        f'to the folder that contains the normal/ and defective/ subfolders.')


def mild_defect_safe_augment(image, rng):
    """Geometry/light variation that should not erase small cracks or chips.

    Deliberately NOT used: random crop, blur, Gaussian noise, cutout/erasing,
    hue or saturation shift. Those can hide or fabricate defect evidence.
    """
    img = image.copy().convert('RGB')
    if rng.random() < 0.50:
        img = ImageOps.mirror(img)
    if rng.random() < 0.50:
        img = ImageOps.flip(img)

    # The source gallery already contains many orientations. Keep rotation mild.
    angle = rng.uniform(-12.0, 12.0)
    fill = tuple(int(x) for x in img.resize((1, 1)).getpixel((0, 0)))
    img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=fill)

    # Mild illumination/camera variation only; preserve colour/staining cues.
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.93, 1.07))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.93, 1.07))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='Train',
                        help='original TRAIN directory; must contain the class folders '
                             '(normal/ and defective/). Case-insensitive.')
    parser.add_argument('--output_dir', default=None,
                        help=f'new output directory. Default: a folder named '
                             f'"{DEFAULT_OUTPUT_NAME}" created next to --input_dir.')
    parser.add_argument('--copies_per_image', type=int, default=2,
                        help='number of augmented copies per original image')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--jpeg_quality', type=int, default=95)
    parser.add_argument('--overwrite', action='store_true',
                        help='delete output_dir first (use deliberately)')
    args = parser.parse_args()

    if args.copies_per_image < 0:
        raise ValueError('--copies_per_image must be >= 0')

    input_dir = resolve_input_dir(args.input_dir)
    # Default: sibling of the train folder, e.g. datasets/pharmacy/Train -> datasets/pharmacy/train_aug
    output_dir = Path(args.output_dir) if args.output_dir else input_dir.parent / DEFAULT_OUTPUT_NAME

    if output_dir.resolve() == input_dir.resolve():
        raise ValueError('output_dir must differ from input_dir; the originals must stay untouched.')
    if input_dir.resolve() in output_dir.resolve().parents:
        raise ValueError(f'output_dir ({output_dir}) sits inside input_dir ({input_dir}). '
                         'Put it outside so the class scan does not pick it up.')
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f'{output_dir} already exists. Pick another output_dir or add --overwrite.')
        shutil.rmtree(output_dir)

    class_dirs = sorted(p for p in input_dir.iterdir() if p.is_dir())
    if not class_dirs:
        raise ValueError(f'No class folders found in {input_dir}')

    print(f'Input : {input_dir}')
    print(f'Output: {output_dir}')
    print(f'Classes found: {", ".join(p.name for p in class_dirs)}')
    print(f'Copies per image: {args.copies_per_image}\n')

    output_dir.mkdir(parents=True)
    rng = random.Random(args.seed)
    total_original, total_augmented = 0, 0

    for class_dir in class_dirs:
        destination = output_dir / class_dir.name
        destination.mkdir()
        files = sorted(p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        if not files:
            print(f'WARNING: no supported image files in {class_dir}')
            continue
        for index, source in enumerate(files):
            # Copy originals exactly, with a stable unique prefix to prevent collisions.
            stem = f'{index:05d}_{source.stem}'
            shutil.copy2(source, destination / f'{stem}_original{source.suffix.lower()}')
            total_original += 1
            with Image.open(source) as opened:
                image = opened.convert('RGB')
                for copy_index in range(args.copies_per_image):
                    augmented = mild_defect_safe_augment(image, rng)
                    # Save JPEG for predictable RGB training input and reasonable disk size.
                    augmented.save(destination / f'{stem}_aug{copy_index + 1}.jpg', quality=args.jpeg_quality)
                    total_augmented += 1
        print(f'{class_dir.name}: {len(files)} originals + {len(files) * args.copies_per_image} augmented '
              f'-> {len(files) * (1 + args.copies_per_image)} images')

    print(f'\nComplete. Originals copied: {total_original}; augmented copies: {total_augmented}; '
          f'total: {total_original + total_augmented}')
    print(f'Point your training loader at: {output_dir}')
    print('Do NOT augment Val or test. Tune the defect threshold on Val and evaluate once on test.')


if __name__ == '__main__':
    main()
