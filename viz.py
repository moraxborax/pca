"""PCA and training-curve visualization helpers for the Train tab."""

from __future__ import annotations

import base64
import json

import cv2
import numpy as np
from numpy.typing import NDArray

from config import ARTIFACTS_DIR, CLASSES, DATA_DIR, IMAGE_SIZE
from pca.pca import pca_reconstruct, pca_residual, pca_transform


def _to_u8_image(arr: NDArray[np.floating], *, symmetric: bool = False) -> NDArray[np.uint8]:
    x = arr.astype(np.float32).reshape(IMAGE_SIZE, IMAGE_SIZE)
    if symmetric:
        lim = float(np.max(np.abs(x))) or 1.0
        x = (x / lim + 1.0) * 127.5
    else:
        lo, hi = float(x.min()), float(x.max())
        if hi <= lo:
            x = np.zeros_like(x)
        else:
            x = (x - lo) / (hi - lo) * 255.0
    return np.clip(x, 0, 255).astype(np.uint8)


def image_to_data_url(image: NDArray[np.uint8], scale: int = 3) -> str:
    if scale != 1:
        image = cv2.resize(
            image,
            (image.shape[1] * scale, image.shape[0] * scale),
            interpolation=cv2.INTER_NEAREST,
        )
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Failed to encode PNG")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def save_training_history(history: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS_DIR / "training_history.json").write_text(json.dumps(history, indent=2))


def load_training_history() -> dict | None:
    path = ARTIFACTS_DIR / "training_history.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def build_pca_viz(
    mean: NDArray[np.float32],
    components: NDArray[np.float32],
    singular_values: NDArray[np.float32],
    n_samples: int,
    residual_threshold: float | None = None,
    n_components_show: int = 8,
) -> dict:
    variance = singular_values.astype(np.float64) ** 2 / max(n_samples - 1, 1)
    total = float(variance.sum()) or 1.0
    ratios = (variance / total).tolist()
    cumulative = np.cumsum(variance / total).tolist()

    eigenimages = []
    for i in range(min(n_components_show, len(components))):
        eigenimages.append(
            {
                "index": i + 1,
                "variance_ratio": ratios[i],
                "image": image_to_data_url(_to_u8_image(components[i], symmetric=True)),
            }
        )

    reconstructions = []
    for class_name in CLASSES:
        paths = sorted((DATA_DIR / class_name).glob("*.png"))
        if not paths:
            continue
        raw = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            continue
        if raw.shape != (IMAGE_SIZE, IMAGE_SIZE):
            raw = cv2.resize(raw, (IMAGE_SIZE, IMAGE_SIZE))
        features = raw.flatten().astype(np.float32) / 255.0
        z = pca_transform(features.reshape(1, -1), mean, components)
        recon = pca_reconstruct(z, mean, components)[0]
        residual = float(pca_residual(features.reshape(1, -1), mean, components)[0])
        residual_map = np.abs(features - recon).reshape(IMAGE_SIZE, IMAGE_SIZE)
        reconstructions.append(
            {
                "class_name": class_name,
                "original": image_to_data_url(raw),
                "reconstructed": image_to_data_url(_to_u8_image(recon)),
                "residual_map": image_to_data_url(_to_u8_image(residual_map)),
                "residual": residual,
            }
        )

    return {
        "mean": image_to_data_url(_to_u8_image(mean)),
        "eigenimages": eigenimages,
        "variance": [
            {"index": i + 1, "ratio": ratios[i], "cumulative": cumulative[i]}
            for i in range(len(ratios))
        ],
        "reconstructions": reconstructions,
        "residual_threshold": residual_threshold,
        "n_components": len(components),
    }


def load_pca_viz() -> dict | None:
    mean_path = ARTIFACTS_DIR / "pca_mean.npy"
    components_path = ARTIFACTS_DIR / "pca_components.npy"
    sv_path = ARTIFACTS_DIR / "singular_values.npy"
    config_path = ARTIFACTS_DIR / "config.json"
    if not (mean_path.exists() and components_path.exists() and sv_path.exists()):
        return None

    mean = np.load(mean_path)
    components = np.load(components_path)
    singular_values = np.load(sv_path)
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    history = load_training_history() or {}
    n_samples = int(history.get("n_samples", max(len(singular_values) + 1, 2)))

    return build_pca_viz(
        mean,
        components,
        singular_values,
        n_samples=n_samples,
        residual_threshold=config.get("residual_threshold"),
    )
