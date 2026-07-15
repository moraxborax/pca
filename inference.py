from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from config import CLASS_TO_INDEX, CLASSES
from pca.pca import pca_residual, pca_transform
from preprocess import frame_to_sample, sample_to_features
from model import Net


@dataclass
class Prediction:
    class_index: int
    class_name: str
    confidence: float
    residual: float = 0.0
    forced_search: bool = False


def predict(
    model: Net,
    mean: NDArray[np.float32],
    components: NDArray[np.float32],
    frame: NDArray[np.uint8],
    device: torch.device = torch.device("cpu"),
    residual_threshold: float | None = None,
) -> Prediction:
    sample = frame_to_sample(frame)
    features = sample_to_features(sample)
    features_2d = features.reshape(1, -1)
    residual = float(pca_residual(features_2d, mean, components)[0])

    if residual_threshold is not None and residual > residual_threshold:
        return Prediction(
            class_index=CLASS_TO_INDEX["search"],
            class_name="search",
            confidence=1.0,
            residual=residual,
            forced_search=True,
        )

    transformed = pca_transform(features_2d, mean, components)
    x = torch.from_numpy(transformed).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        confidence, class_index = torch.max(probs, dim=1)
        idx = int(class_index.item())

    return Prediction(
        class_index=idx,
        class_name=CLASSES[idx],
        confidence=float(confidence.item()),
        residual=residual,
        forced_search=False,
    )
