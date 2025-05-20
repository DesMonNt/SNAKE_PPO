import random
import numpy as np


class SnakeEnv:
    def __init__(self, grid_size: int = 10, num_food: int = 3, seed: int = None):
        self.grid_size = grid_size
        self.num_food = num_food
        self.random = random.Random(seed)
        np.random.seed(seed)
        self.just_reset = False
        self.reset()

    def reset(self):
        self.snake = [np.array([self.grid_size // 2, self.grid_size // 2])]
        self.direction = np.array(self.random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)]))
        self.food = set()
        self._spawn_food()
        self.prev_distance = self._distance_to_nearest_food()
        self.just_reset = True

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

        new_head = self.snake[0] + self.direction
        reward = -0.01
        done = False

        if not (0 <= new_head[0] < self.grid_size and 0 <= new_head[1] < self.grid_size):
            reward = -1.0
            done = True

            return self._get_obs(), reward, done

        if any(np.array_equal(new_head, part) for part in self.snake):
            reward = -1.0
            done = True

            return self._get_obs(), reward, done

        self.snake.insert(0, new_head.copy())

        if tuple(new_head) in self.food:
            reward = 1.0
            self.food.remove(tuple(new_head))
            self._spawn_food()
        else:
            self.snake.pop()

        new_distance = self._distance_to_nearest_food()
        delta = self.prev_distance - new_distance

        if abs(delta) > 0.1:
            reward += 0.1 * np.sign(delta)

        self.prev_distance = new_distance

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

    def _distance_to_nearest_food(self):
        return min(np.sum(np.abs(self.snake[0] - np.array(f))) for f in self.food)