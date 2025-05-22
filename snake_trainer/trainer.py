import os
import torch
import numpy as np
from tqdm import trange
from snake_agent import PPOAgent, SnakePolicyNetwork
from multiprocessing import Process, Pipe
from snake_trainer.rollout_buffer import RolloutBuffer
from snake_trainer.utils import worker


class Trainer:
    def __init__(
        self,
        env_fn,
        num_actions=4,
        num_episodes=1000,
        rollout_len=2048,
        update_epochs=4,
        batch_size=64,
        save_path="checkpoints/ppo_snake.pt",
        num_envs=4,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.env_fn = env_fn
        self.model = SnakePolicyNetwork(num_actions=num_actions).to(self.device)
        self.agent = PPOAgent(self.model)

        self.num_actions = num_actions
        self.num_episodes = num_episodes
        self.rollout_len = rollout_len
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        self.save_path = save_path

        self.reward_log = []
        self.best_reward = -float("inf")
        self.num_envs = num_envs

        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(num_envs)])
        self.ps = [Process(target=worker, args=(work_remote, self.env_fn)) for work_remote in self.work_remotes]

        for p in self.ps:
            p.start()

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def train(self):
        for remote in self.remotes:
            remote.send(("reset", None))

        obs_batch = [remote.recv() for remote in self.remotes]
        progress = trange(self.num_episodes, desc="Training", ncols=100)
        episode_reward_tracker = [0.0 for _ in range(self.num_envs)]

        for ep in progress:
            buffer = RolloutBuffer()

            for _ in range(self.rollout_len // self.num_envs):
                actions, log_probs, values = [], [], []
                states = [obs.to(self.device) for obs in obs_batch]

                for s in states:
                    a, log_p, v = self.agent.select_action(s)
                    actions.append(a)
                    log_probs.append(log_p)
                    values.append(v)

                for i, remote in enumerate(self.remotes):
                    remote.send(("step", actions[i]))

                results = [remote.recv() for remote in self.remotes]

                for i, (next_obs, reward, done) in enumerate(results):
                    buffer.add(states[i], actions[i], log_probs[i], values[i], reward, done)
                    episode_reward_tracker[i] += reward
                    obs_batch[i] = next_obs

                    if done:
                        self.reward_log.append(episode_reward_tracker[i])
                        episode_reward_tracker[i] = 0.0

            advantages, returns = self.agent.compute_gae(buffer.rewards, buffer.values, buffer.dones)
            self.agent.update(
                (buffer.states, buffer.actions, buffer.log_probs, returns, advantages),
                epochs=self.update_epochs,
                batch_size=self.batch_size
            )

            if len(self.reward_log) >= 100:
                avg_reward = np.mean(self.reward_log[-100:])
                progress.set_description(f"Ep {ep:5d} | AvgR: {avg_reward:.2f}")

                if avg_reward > self.best_reward:
                    self.best_reward = avg_reward
                    torch.save(self.model.state_dict(), self.save_path)

        for remote in self.remotes:
            remote.send(("close", None))

        for p in self.ps:
            p.join()
