import torch.nn as nn

from config import NUM_CLASSES


class Net(nn.Module):
    def __init__(self, components: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(components, 64),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)
