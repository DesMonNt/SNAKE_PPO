from snake_game import SnakeEnv
from .utils import obs_to_tensor


class SnakeWrapper:
    def __init__(self, env: SnakeEnv):
        self.env = env
        self.grid_size = env.grid_size
        self.num_actions = 4

    def reset(self):
        obs = self.env.reset()
        return obs_to_tensor(obs, grid_size=self.grid_size)

    def step(self, action):
        obs, reward, done = self.env.step(action)
        return obs_to_tensor(obs, grid_size=self.grid_size), reward, done