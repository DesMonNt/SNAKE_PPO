import torch
from snake_game import SnakeEnv, SnakeRenderer
from snake_agent import SnakeCNNPolicy, SnakeWrapper

def evaluate_agent(model_path="checkpoints/snake_700.pt", grid_size=10, fps=10):
    env = SnakeWrapper(SnakeEnv(grid_size=grid_size, num_food=5))
    model = SnakeCNNPolicy(grid_size=grid_size, num_actions=4)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    state = env.reset()
    done = False
    total_reward = 0

    renderer = SnakeRenderer(env.env, fps=fps)

    while not done:
        renderer.render()
        with torch.no_grad():
            q_values = model(state)
            action = q_values.argmax().item()

        state, reward, done = env.step(action)
        total_reward += reward

    renderer.close()

if __name__ == "__main__":
    evaluate_agent()
