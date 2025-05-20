import sys
import pygame
import time
from snake_game import *


def main():
    env = SnakeEnv(grid_size=10, num_food=5)
    renderer = SnakeRenderer(env, fps=60)
    env.reset()

    action = 3
    running = True
    tick_duration = 0.15
    last_step_time = time.time()

    while running:
        current_time = time.time()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0
                elif event.key == pygame.K_LEFT:
                    action = 1
                elif event.key == pygame.K_DOWN:
                    action = 2
                elif event.key == pygame.K_RIGHT:
                    action = 3

        if current_time - last_step_time >= tick_duration:
            obs, reward, done = env.step(action)
            last_step_time = current_time

            if done:
                time.sleep(1)
                env.reset()

        renderer.render()

    renderer.close()
    pygame.quit()
    sys.exit()


if __name__ == '__main__':
    main()