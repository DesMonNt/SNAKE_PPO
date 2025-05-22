import torch
import torch.nn as nn


class SnakePolicyNetwork(nn.Module):
    def __init__(
        self,
        num_actions: int,
        in_channels: int = 5,
        cnn_channels: int = 64,
        num_heads: int = 4,
        mlp_hidden_dim: int = 256,
        mlp_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, cnn_channels))
        nn.init.normal_(self.cls_token, std=0.02)

        self.attn = nn.MultiheadAttention(
            embed_dim=cnn_channels,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        mlp_layers_list = [nn.Linear(cnn_channels, mlp_hidden_dim), nn.ReLU()]
        for _ in range(mlp_layers - 1):
            mlp_layers_list.extend([
                nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
        self.mlp = nn.Sequential(*mlp_layers_list)

        self.policy_head = nn.Linear(mlp_hidden_dim, num_actions)
        self.value_head = nn.Linear(mlp_hidden_dim, 1)

    def forward(self, x):
        x = self.cnn(x)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)

        cls = self.cls_token.expand(B, 1, -1)
        x = torch.cat((cls, x), dim=1)

        x, _ = self.attn(x, x, x)
        x = x[:, 0]
        x = self.mlp(x[:, 0])

        return self.policy_head(x), self.value_head(x).squeeze(-1)
