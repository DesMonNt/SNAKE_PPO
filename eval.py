import torch
from snake_game import SnakeEnv, SnakeRenderer
from snake_agent import SnakeCNNPolicy, SnakeWrapper
from torch.distributions import Categorical


def evaluate_agent(model_path="checkpoints/snake_2500.pt", grid_size=10, fps=10):
    env = SnakeWrapper(SnakeEnv(grid_size=grid_size, num_food=5))
    model = SnakeCNNPolicy(grid_size=grid_size, num_actions=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    renderer = SnakeRenderer(env.env, fps=fps)

    try:
        while True:
            state = env.reset()
            done = False
            total_reward = 0

            while not done:
                renderer.render()

                with torch.no_grad():
                    q_values = model(state)
                    probs = torch.softmax(q_values, dim=-1)
                    action = Categorical(probs).sample().item()

                state, reward, done = env.step(action)
                total_reward += reward
    finally:
        renderer.close()

if __name__ == "__main__":
    evaluate_agent()
