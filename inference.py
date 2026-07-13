from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from config import CLASSES
from pca.pca import pca_transform
from preprocess import frame_to_sample, sample_to_features
from model import Net


@dataclass
class Prediction:
    class_index: int
    class_name: str
    confidence: float


def predict(
    model: Net,
    mean: NDArray[np.float32],
    components: NDArray[np.float32],
    frame: NDArray[np.uint8],
    device: torch.device = torch.device("cpu"),
) -> Prediction:
    sample = frame_to_sample(frame)
    features = sample_to_features(sample)
    transformed = pca_transform(features.reshape(1, -1), mean, components)
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
    )
