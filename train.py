import torch
import numpy as np
from tqdm import trange

from snake_game import SnakeEnv
from snake_agent import SnakePPOModel, PPOAgent, SnakeWrapper


def train_ppo(
    grid_size=10,
    num_actions=4,
    num_episodes=10000,
    rollout_len=2048,
    update_epochs=4,
    batch_size=128,
    save_path="ppo_snake.pt",
    checkpoint_every=100
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = SnakeWrapper(SnakeEnv(grid_size=grid_size, num_food=5))
    model = SnakePPOModel(num_actions=num_actions).to(device)
    agent = PPOAgent(model)

    best_reward = -float("inf")
    reward_log = []

    obs = env.reset()
    episode_reward = 0

    rollout_buffer = {
        "states": [], "actions": [], "log_probs": [],
        "values": [], "rewards": [], "dones": []
    }

    progress = trange(num_episodes, desc="Training", ncols=100)
    for ep in progress:
        for _ in range(rollout_len):
            state = obs.to(device)
            action, log_prob, value = agent.select_action(state)

            next_obs, reward, done = env.step(action)
            rollout_buffer["states"].append(state)
            rollout_buffer["actions"].append(action)
            rollout_buffer["log_probs"].append(log_prob)
            rollout_buffer["values"].append(value)
            rollout_buffer["rewards"].append(reward)
            rollout_buffer["dones"].append(done)

            obs = next_obs
            episode_reward += reward

            if done:
                obs = env.reset()
                reward_log.append(episode_reward)
                episode_reward = 0

        advantages, returns = agent.compute_gae(
            rollout_buffer["rewards"],
            rollout_buffer["values"],
            rollout_buffer["dones"]
        )

        rollouts = (
            rollout_buffer["states"],
            rollout_buffer["actions"],
            rollout_buffer["log_probs"],
            returns,
            advantages
        )
        agent.update(rollouts, epochs=update_epochs, batch_size=batch_size)
        for k in rollout_buffer:
            rollout_buffer[k] = []

        if len(reward_log) >= 100:
            avg_reward = np.mean(reward_log[-100:])
            progress.set_description(f"Ep {ep:5d} | AvgR: {avg_reward:.2f}")

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(model.state_dict(), save_path)

        if ep % checkpoint_every == 0 and ep > 0:
            torch.save(model.state_dict(), f"ppo_snake_{ep}.pt")

    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    train_ppo()
