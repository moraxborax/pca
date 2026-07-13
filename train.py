import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from artifacts import save_artifacts
from config import N_COMPONENTS
from dataset import load_dataset
from model import Net
from pca.pca import pca, pca_transform

def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
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

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_samples += x.size(0)

    avg_loss = total_loss / total_samples
    accuracy = total_correct / total_samples

    print(f"Loss: {avg_loss:.4f}  Accuracy: {accuracy:.2%}")
    return avg_loss


def analyze_variance(singular_values: np.ndarray, n_samples: int) -> None:
    variance = singular_values**2 / max(n_samples - 1, 1)
    total_variance = variance.sum()
    cumulative = np.cumsum(variance)

    print("\nPCA variance analysis (first 128 components):")
    print(f"{'Comp':>4}  {'Var':>12}  {'Ratio':>8}  {'Cumulative':>10}")
    for i, (v, c) in enumerate(zip(variance, cumulative), start=1):
        ratio = v / total_variance if total_variance > 0 else 0.0
        cum_ratio = c / total_variance if total_variance > 0 else 0.0
        if i <= 10 or i % 16 == 0 or i == len(variance):
            print(f"{i:4d}  {v:12.6f}  {ratio:8.4f}  {cum_ratio:10.4f}")
    if total_variance > 0:
        print(f"\nTotal variance captured by {len(variance)} components: {cumulative[-1] / total_variance:.2%}")


def main() -> None:
    device = torch.device("cpu")
    X, y = load_dataset()
    if len(X) == 0:
        raise SystemExit("No training data found. Collect samples in label mode first.")

    print(f"Loaded {len(X)} samples with {X.shape[1]} features")

    n_components = min(N_COMPONENTS, X.shape[0] - 1, X.shape[1])
    mean, components, singular_values = pca(X, n_components)
    analyze_variance(singular_values, len(X))

    X_pca = pca_transform(X, mean, components)
    print(f"PCA transformed shape: {X_pca.shape}")

    indices = np.arange(len(X))
    np.random.default_rng(42).shuffle(indices)
    split = int(0.8 * len(X))
    train_idx, val_idx = indices[:split], indices[split:]

    X_train = torch.from_numpy(X_pca[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    X_val = torch.from_numpy(X_pca[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32)

    model = Net(n_components).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 30
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_epoch(model, train_loader, optimizer, criterion, device)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, labels in val_loader:
            x = x.to(device)
            labels = labels.to(device)
            logits = model(x)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)
    if total > 0:
        print(f"\nValidation accuracy: {correct / total:.2%}")

    save_artifacts(model, mean, components, singular_values, n_components)
    print("\nSaved artifacts to artifacts/")


if __name__ == "__main__":
    main()
