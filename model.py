import torch.nn as nn


class Net(nn.Module):
    def __init__(self, components: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(components, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 5),
        )

    def forward(self, x):
        return self.net(x)
