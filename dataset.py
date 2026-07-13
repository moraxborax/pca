from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from config import CLASS_TO_INDEX, CLASSES, DATA_DIR
from preprocess import sample_to_features


def ensure_class_dirs() -> None:
    for class_name in CLASSES:
        (DATA_DIR / class_name).mkdir(parents=True, exist_ok=True)


def save_sample(class_name: str, sample: NDArray[np.uint8]) -> Path:
    if class_name not in CLASS_TO_INDEX:
        raise ValueError(f"Unknown class: {class_name}")
    ensure_class_dirs()
    filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
    path = DATA_DIR / class_name / filename
    cv2.imwrite(str(path), sample)
    return path


def class_counts() -> dict[str, int]:
    ensure_class_dirs()
    return {
        class_name: len(list((DATA_DIR / class_name).glob("*.png")))
        for class_name in CLASSES
    }


def clear_dataset() -> dict[str, int]:
    """Delete all PNG samples. Returns counts after clearing (all zeros)."""
    ensure_class_dirs()
    for class_name in CLASSES:
        for image_path in (DATA_DIR / class_name).glob("*.png"):
            image_path.unlink(missing_ok=True)
    return class_counts()


def load_dataset() -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    ensure_class_dirs()
    features: list[NDArray[np.float32]] = []
    labels: list[int] = []

    for class_name, class_index in CLASS_TO_INDEX.items():
        for image_path in sorted((DATA_DIR / class_name).glob("*.png")):
            image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            features.append(sample_to_features(image))
            labels.append(class_index)

    if not features:
        return np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=np.int64)

    return np.stack(features), np.array(labels, dtype=np.int64)
