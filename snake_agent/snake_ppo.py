import torch.nn as nn


class SnakePPOModel(nn.Module):
    def __init__(self, num_actions: int):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.mlp = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, num_actions)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.cnn(x)
        x = self.mlp(x)

        return self.policy_head(x), self.value_head(x).squeeze(-1)
