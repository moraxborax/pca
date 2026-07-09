import numpy as np
import torch
import torch.nn as nn
from pca.pca import pca


class Net(nn.Module):
    def __init__(self, components: int):
        super().__init__()
        self.net = nn.Sequential(
            # nn.Linear(components, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            nn.Linear(components, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(Net.parameters(), lr=1e-3)
# can also try sgd or adamw later

def train(
    model: nn.Module,
    dataloader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(x)

        # Compute loss
        loss = criterion(logits, y)

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Loss: {avg_loss:.4f}  Accuracy: {accuracy:.2%}")

    return avg_loss