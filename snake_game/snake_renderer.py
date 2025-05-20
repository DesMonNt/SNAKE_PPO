import pygame
from .snake_env import SnakeEnv


class SnakeRenderer:
    def __init__(self, env: SnakeEnv, tile_size: int = 30, fps: int = 10):
        pygame.init()
        pygame.display.set_caption("Snake RL")

        self.env = env
        self.tile_size = tile_size
        self.fps = fps
        self.grid_size = env.grid_size
        self.screen_size = self.grid_size * tile_size
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        self.clock = pygame.time.Clock()

    def render(self):
        self.screen.fill((0, 0, 0))

        for fx, fy in self.env.food:
            pygame.draw.rect(self.screen, (255, 0, 0), (fy * self.tile_size, fx * self.tile_size, self.tile_size, self.tile_size))

        for idx, (x, y) in enumerate(self.env.snake):
            color = (0, 255, 0) if idx > 0 else (0, 200, 255)
            pygame.draw.rect(self.screen, color, (y * self.tile_size, x * self.tile_size, self.tile_size, self.tile_size))

        for x in range(0, self.screen_size, self.tile_size):
            pygame.draw.line(self.screen, (40, 40, 40), (x, 0), (x, self.screen_size))

        for y in range(0, self.screen_size, self.tile_size):
            pygame.draw.line(self.screen, (40, 40, 40), (0, y), (self.screen_size, y))

        pygame.display.flip()
        self.clock.tick(self.fps)

    def close(self):
        pygame.quit()
