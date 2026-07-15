import json

import numpy as np
import torch
from numpy.typing import NDArray

from config import ARTIFACTS_DIR, CLASSES, IMAGE_SIZE
from model import Net


def artifacts_exist() -> bool:
    return (ARTIFACTS_DIR / "model.pth").exists() and (ARTIFACTS_DIR / "pca_mean.npy").exists()


def save_artifacts(
    model: Net,
    mean: NDArray[np.float32],
    components: NDArray[np.float32],
    singular_values: NDArray[np.float32],
    n_components: int,
    residual_threshold: float,
) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    np.save(ARTIFACTS_DIR / "pca_mean.npy", mean)
    np.save(ARTIFACTS_DIR / "pca_components.npy", components)
    np.save(ARTIFACTS_DIR / "singular_values.npy", singular_values)
    torch.save(model.state_dict(), ARTIFACTS_DIR / "model.pth")
    config = {
        "image_size": IMAGE_SIZE,
        "n_components": n_components,
        "classes": CLASSES,
        "residual_threshold": residual_threshold,
    }
    (ARTIFACTS_DIR / "config.json").write_text(json.dumps(config, indent=2))


def load_artifacts(
    device: torch.device = torch.device("cpu"),
) -> tuple[Net, NDArray[np.float32], NDArray[np.float32], dict] | None:
    if not artifacts_exist():
        return None

    config = json.loads((ARTIFACTS_DIR / "config.json").read_text())
    if config.get("classes") != CLASSES:
        # Old checkpoints (e.g. 5-class) are incompatible with the current label set.
        return None

    n_components = int(config["n_components"])
    mean = np.load(ARTIFACTS_DIR / "pca_mean.npy")
    components = np.load(ARTIFACTS_DIR / "pca_components.npy")

    model = Net(n_components, num_classes=len(CLASSES))
    model.load_state_dict(
        torch.load(ARTIFACTS_DIR / "model.pth", map_location=device, weights_only=True)
    )
    model.to(device)
    model.eval()

    return model, mean, components, config
