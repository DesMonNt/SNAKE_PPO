import torch
from torch import nn


def obs_to_tensor(obs, grid_size):
    tensor = torch.zeros((5, grid_size, grid_size), dtype=torch.float32)

    hx, hy = obs["head"]
    tensor[0, hx, hy] = 1.0

    for x, y in obs["snake"][1:]:
        tensor[1, x, y] = 1.0

    for fx, fy in obs["food"]:
        tensor[2, fx, fy] = 1.0

    tensor[3] = torch.linspace(0.01, 0.99, grid_size).repeat(grid_size, 1)
    tensor[4] = torch.linspace(0.01, 0.99, grid_size).unsqueeze(1).repeat(1, grid_size)

    return tensor

def init_weights(module):
    if isinstance(module, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, nonlinearity='relu')

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)