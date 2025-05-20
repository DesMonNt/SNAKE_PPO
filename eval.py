import torch
from torch.distributions import Categorical
from snake_game import SnakeEnv, SnakeRenderer
from snake_agent import SnakePPOModel, SnakeWrapper


def evaluate_agent(model_path="ppo_snake.pt", grid_size=10, fps=10, deterministic=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeWrapper(SnakeEnv(grid_size=grid_size, num_food=5))
    model = SnakePPOModel(num_actions=4).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    renderer = SnakeRenderer(env.env, fps=fps)

    try:
        while True:
            state = env.reset().to(device)
            done = False
            total_reward = 0

            while not done:
                renderer.render()

                with torch.no_grad():
                    logits, _ = model(state)
                    dist = Categorical(logits=logits)

                    if deterministic:
                        action = torch.argmax(logits, dim=-1).item()
                    else:
                        action = dist.sample().item()

                state, reward, done = env.step(action)
                state = state.to(device)
                total_reward += reward
    finally:
        renderer.close()


if __name__ == "__main__":
    evaluate_agent(deterministic=True)
