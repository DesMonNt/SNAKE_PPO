from snake_agent.utils import obs_to_tensor
from snake_game import SnakeEnv


class SnakeWrapper:
    def __init__(self, env: SnakeEnv, transform=None):
        self.env = env
        self.grid_size = env.grid_size
        self.num_actions = 4
        self.transform = transform if transform is not None else lambda x: obs_to_tensor(x, grid_size=self.grid_size)

    def reset(self):
        obs = self.env.reset()
        return self.transform(obs) if self.transform else obs

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return self.transform(obs) if self.transform else obs, reward, done