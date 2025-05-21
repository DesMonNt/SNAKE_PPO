import random
import numpy as np


class SnakeEnv:
    def __init__(self, grid_size: int = 10, num_food: int = 3, seed: int = None):
        self.grid_size = grid_size
        self.num_food = num_food
        self.random = random.Random(seed)
        np.random.seed(seed)

        self.food = set()
        self.just_reset = False
        self.reset()

    def reset(self):
        self.snake = [np.array([self.grid_size // 2, self.grid_size // 2])]
        self.direction = np.array(self.random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)]))
        self.food.clear()
        self._spawn_food()

        self.just_reset = True
        self.steps_since_last_food = 0
        self.total_steps = 0

        return self._get_obs()

    def step(self, action: int):
        dir_map = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, -1]),  # Left
            2: np.array([1, 0]),   # Down
            3: np.array([0, 1]),   # Right
        }
        new_dir = dir_map[action]

        if not np.array_equal(new_dir, -self.direction) or self.just_reset:
            self.direction = new_dir

        self.just_reset = False
        self.total_steps += 1

        new_head = self.snake[0] + self.direction
        reward = -0.01
        done = False

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = -1.0
            done = True
        elif any(np.array_equal(new_head, part) for part in self.snake):
            reward = -1.0
            done = True
        else:
            self.snake.insert(0, new_head.copy())

            if tuple(new_head) in self.food:
                reward += 1.0

                self.food.remove(tuple(new_head))
                self._spawn_food()
            else:
                self.snake.pop()

        if done and self.total_steps <= 10:
            reward -= 1.0

        if done and self.total_steps <= 5:
            reward -= 1.0

        return self._get_obs(), reward, done

    def seed(self, seed=None):
        self.random.seed(seed)
        np.random.seed(seed)

    def _spawn_food(self):
        needed = self.num_food - len(self.food)
        if needed <= 0:
            return

        occupied = {tuple(p) for p in self.snake} | self.food
        available = [(i, j) for i in range(self.grid_size) for j in range(self.grid_size)
                     if (i, j) not in occupied]

        new_food = self.random.sample(available, min(len(available), needed))
        self.food.update(new_food)

    def _get_obs(self):
        return {
            "snake": [tuple(p) for p in self.snake],
            "food": self.food.copy(),
            "head": tuple(self.snake[0]),
        }