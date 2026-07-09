from pca.pca import pca_transform
from train import Net
import numpy as np
from numpy.typing import NDArray
import torch

def inference(model: Net, mean: NDArray[np.float32], components: NDArray[np.float32], data: NDArray[np.uint8]):
    flattened_data = data.flatten().astype(np.float32) / 255.0
    transformed_data = torch.from_numpy(pca_transform(flattened_data, mean, components)).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        logits = model(transformed_data)
        return torch.argmax(logits, dim=1).item()

ACTIONS = ["up", "down", "left", "right", "search"]

