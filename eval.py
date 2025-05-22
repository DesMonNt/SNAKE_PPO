import sys
import time
import pygame
import torch
from torch.distributions import Categorical
from snake_game import SnakeEnv, SnakeRenderer
from snake_agent import Wrapper, SnakePolicyNetwork


def evaluate_agent(model_path="ppo_snake.pt", grid_size=10, num_food=5, fps=10, deterministic=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SnakePolicyNetwork(num_actions=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    env = Wrapper(SnakeEnv(grid_size=grid_size, num_food=num_food))
    renderer = SnakeRenderer(env.env, fps=fps)
    state = env.reset().to(device)

    running = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        with torch.no_grad():
            logits, _ = model(state.unsqueeze(0))
            dist = Categorical(logits=logits)

            if deterministic:
                action = torch.argmax(logits, dim=-1).item()
            else:
                action = dist.sample().item()

        state, reward, done = env.step(action)
        state = state.to(device)

        if done:
            time.sleep(1)
            env.reset()

        renderer.render()

    renderer.close()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    evaluate_agent(
        grid_size=10,
        num_food=5,
        deterministic=False
    )
