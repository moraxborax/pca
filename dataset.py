from __future__ import annotations

from datetime import datetime
from pathlib import Path
import shutil

import cv2
import numpy as np
from numpy.typing import NDArray

from config import (
    CLASS_TO_INDEX,
    CLASSES,
    DATA_DIR,
    DATASET_SPLITS,
    TEST_DIR,
    TRAIN_DIR,
)
from preprocess import sample_to_features


def _validate_split(split: str) -> str:
    if split not in DATASET_SPLITS:
        raise ValueError(f"Invalid split: {split}. Expected one of {DATASET_SPLITS}")
    return split


def split_dir(split: str) -> Path:
    return DATA_DIR / _validate_split(split)


def migrate_legacy_layout() -> None:
    """Move data/{class}/*.png into data/train/{class}/ if present."""
    moved_any = False
    for class_name in CLASSES:
        legacy = DATA_DIR / class_name
        if not legacy.is_dir():
            continue
        pngs = list(legacy.glob("*.png"))
        if not pngs:
            continue
        dest = TRAIN_DIR / class_name
        dest.mkdir(parents=True, exist_ok=True)
        for path in pngs:
            target = dest / path.name
            if target.exists():
                target = dest / f"legacy_{path.name}"
            shutil.move(str(path), str(target))
            moved_any = True
        # Remove empty legacy class dir
        try:
            legacy.rmdir()
        except OSError:
            pass
    if moved_any:
        ensure_class_dirs()


def ensure_class_dirs() -> None:
    for split in DATASET_SPLITS:
        for class_name in CLASSES:
            (DATA_DIR / split / class_name).mkdir(parents=True, exist_ok=True)


def save_sample(
    class_name: str,
    sample: NDArray[np.uint8],
    split: str = "train",
) -> Path:
    if class_name not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown class: {class_name}")
    ensure_class_dirs()
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path = split_dir(split) / class_name / filename
    cv2.imwrite(str(path), sample)
    return path


def class_counts(split: str | None = None) -> dict[str, int] | dict[str, dict[str, int]]:
    ensure_class_dirs()
    migrate_legacy_layout()
    ensure_class_dirs()

    def counts_for(s: str) -> dict[str, int]:
        root = split_dir(s)
        return {
            class_name: len(list((root / class_name).glob("*.png")))
            for class_name in CLASSES
        }

    if split is not None:
        return counts_for(split)
    return {s: counts_for(s) for s in DATASET_SPLITS}


def clear_dataset(split: str | None = None) -> dict[str, dict[str, int]]:
    """Delete PNG samples. split=None clears train and test."""
    ensure_class_dirs()
    migrate_legacy_layout()
    targets = DATASET_SPLITS if split is None else (_validate_split(split),)
    for s in targets:
        for class_name in CLASSES:
            for image_path in (split_dir(s) / class_name).glob("*.png"):
                image_path.unlink(missing_ok=True)
    return class_counts()  # type: ignore[return-value]


def load_dataset(split: str = "train") -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    ensure_class_dirs()
    migrate_legacy_layout()
    ensure_class_dirs()
    root = split_dir(split)
    features: list[NDArray[np.float32]] = []
    labels: list[int] = []

    for class_name, class_index in CLASS_TO_INDEX.items():
        for image_path in sorted((root / class_name).glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            features.append(sample_to_features(image))
            labels.append(class_index)

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(features), np.array(labels, dtype=np.int64)


# Re-export paths for callers
__all__ = [
    "TRAIN_DIR",
    "TEST_DIR",
    "ensure_class_dirs",
    "migrate_legacy_layout",
    "save_sample",
    "class_counts",
    "clear_dataset",
    "load_dataset",
    "split_dir",
]
