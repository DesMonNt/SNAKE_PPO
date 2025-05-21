import torch.nn as nn


class SnakePPOModel(nn.Module):
    def __init__(
        self,
        grid_size: int,
        num_actions: int,
        cnn_channels: int = 64,
        num_heads: int = 4,
        mlp_hidden_dim: int = 256,
        mlp_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=cnn_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        mlp_modules = [nn.Linear(cnn_channels * grid_size * grid_size, mlp_hidden_dim), nn.ReLU()]

        for _ in range(mlp_layers - 1):
            mlp_modules.extend([
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])

        self.mlp = nn.Sequential(*mlp_modules)
        self.policy_head = nn.Linear(mlp_hidden_dim, num_actions)
        self.value_head = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, x):
        x = self.cnn(x)

        B, C, H, W = x.shape

        x = x.view(B, C, H * W).permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.flatten(start_dim=1)
        x = self.mlp(x)

        return self.policy_head(x), self.value_head(x).squeeze(-1)
