"""
dataset_utils.py — Dataset Utility Functions
=============================================
Shared helpers used across the three notebooks.
"""

import os
import shutil
import random
import numpy as np
from PIL import Image, ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True

CLASSES = [
    'healthy_forest',
    'deforested',
    'forest_fire',
    'flooded_forest',
    'diseased_forest',
]

IMG_SIZE = (224, 224)


def standardize_images(class_dir: str, img_size: tuple = IMG_SIZE) -> tuple[int, int]:
    """
    Validate, convert to RGB, resize, and re-save all images in a directory.

    Returns:
        (valid_count, removed_count)
    """
    if not os.path.exists(class_dir):
        return 0, 0

    files   = os.listdir(class_dir)
    valid   = 0
    removed = 0

    for fname in tqdm(files, desc=f'  {os.path.basename(class_dir)}', leave=False):
        fpath = os.path.join(class_dir, fname)
        try:
            with Image.open(fpath) as img:
                img = img.convert('RGB')
                img = img.resize(img_size, Image.LANCZOS)
                new_name = f'{os.path.basename(class_dir)}_{valid:04d}.jpg'
                new_path = os.path.join(class_dir, new_name)
                img.save(new_path, 'JPEG', quality=90)
                if fpath != new_path:
                    os.remove(fpath)
                valid += 1
        except Exception:
            try:
                os.remove(fpath)
            except OSError:
                pass
            removed += 1

    return valid, removed


def split_dataset(
    raw_dir: str,
    base_dir: str,
    classes: list,
    train: float = 0.70,
    val: float   = 0.15,
    test: float  = 0.15,
    seed: int    = 42,
) -> dict:
    """
    Copy images from raw_dir/<class>/ into base_dir/{train,val,test}/<class>/.

    Returns:
        dict of {split: {class: count}}
    """
    assert abs(train + val + test - 1.0) < 1e-6, 'Splits must sum to 1.0'

    counts = {'train': {}, 'val': {}, 'test': {}}

    for cls in classes:
        src = os.path.join(raw_dir, cls)
        if not os.path.exists(src):
            print(f'⚠️  Missing: {src}')
            continue

        imgs = sorted(f for f in os.listdir(src) if f.endswith('.jpg'))
        random.seed(seed)
        random.shuffle(imgs)

        n1 = int(len(imgs) * train)
        n2 = int(len(imgs) * val)

        splits = {
            'train': imgs[:n1],
            'val'  : imgs[n1: n1 + n2],
            'test' : imgs[n1 + n2:],
        }

        for split_name, split_files in splits.items():
            dst = os.path.join(base_dir, split_name, cls)
            os.makedirs(dst, exist_ok=True)
            for f in split_files:
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
            counts[split_name][cls] = len(split_files)

    return counts


def compute_class_weights(train_dir: str, classes: list) -> dict:
    """
    Compute class weights to handle imbalanced datasets.
    Returns a dict mapping class index to weight.
    """
    counts = []
    for cls in classes:
        p = os.path.join(train_dir, cls)
        counts.append(len(os.listdir(p)) if os.path.exists(p) else 0)

    total   = sum(counts)
    n_cls   = len(classes)
    weights = {i: total / (n_cls * c) if c > 0 else 1.0 for i, c in enumerate(counts)}
    return weights


def count_dataset(base_dir: str, classes: list) -> None:
    """Print a summary table of the dataset splits."""
    splits = ['train', 'val', 'test']
    print(f'\n{"Class":<22}', end='')
    for s in splits:
        print(f'{s.capitalize():>8}', end='')
    print(f'{"Total":>8}')
    print('-' * 54)

    for cls in classes:
        print(f'{cls:<22}', end='')
        row_total = 0
        for s in splits:
            p = os.path.join(base_dir, s, cls)
            n = len(os.listdir(p)) if os.path.exists(p) else 0
            print(f'{n:>8}', end='')
            row_total += n
        print(f'{row_total:>8}')

    print('-' * 54)
    print(f'{"TOTAL":<22}', end='')
    grand = 0
    for s in splits:
        col_total = sum(
            len(os.listdir(os.path.join(base_dir, s, c)))
            for c in classes
            if os.path.exists(os.path.join(base_dir, s, c))
        )
        print(f'{col_total:>8}', end='')
        grand += col_total
    print(f'{grand:>8}\n')
