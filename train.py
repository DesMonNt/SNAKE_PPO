import torch
import numpy as np
import os
from tqdm import trange
from snake_game import SnakeEnv
from snake_agent import DQNAgent, SnakeCNNPolicy, SnakeWrapper


def train_dqn(
    grid_size=10,
    num_episodes=10000,
    target_update_freq=20,
    checkpoint_freq=100,
    save_path="snake_policy.pt",
    checkpoint_dir="checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    env = SnakeWrapper(SnakeEnv(grid_size=grid_size, num_food=5))
    policy = SnakeCNNPolicy(grid_size=grid_size, num_actions=4)
    target = SnakeCNNPolicy(grid_size=grid_size, num_actions=4)
    target.load_state_dict(policy.state_dict())

    agent = DQNAgent(policy, target)

    best_reward = -float('inf')
    reward_log = []

    progress = trange(1, num_episodes + 1, desc="Training", ncols=100)

    for episode in progress:
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state, deterministic=False)
            next_state, reward, done = env.step(action)
            agent.store_transition(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward

        reward_log.append(total_reward)

        if episode % target_update_freq == 0:
            agent.update_target()

        if episode % 10 == 0:
            avg_reward = np.mean(reward_log[-100:])
            epsilon = agent.compute_epsilon()
            progress.set_description(f"Ep {episode:5d} | AvgR: {avg_reward:+.3f} | ε: {epsilon:.3f}")
            progress.refresh()

            if avg_reward > best_reward:
                best_reward = avg_reward
                torch.save(policy.state_dict(), save_path)

        if episode % checkpoint_freq == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"snake_{episode}.pt")
            torch.save(policy.state_dict(), checkpoint_path)

    torch.save(policy.state_dict(), save_path)


if __name__ == '__main__':
    train_dqn()