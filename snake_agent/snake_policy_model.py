import torch
import torch.nn as nn
import torch.nn.functional as F


class SnakeCNNPolicy(nn.Module):
    def __init__(self, grid_size, num_actions):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * grid_size * grid_size, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.cnn(x)
        return self.mlp(x)
