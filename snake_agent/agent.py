import math
import torch
import random
from collections import deque


class DQNAgent:
    def __init__(self,
                 policy_net,
                 target_net,
                 lr=1e-4,
                 gamma=0.99,
                 epsilon_start=1.0,
                 epsilon_end=0.1,
                 epsilon_decay=5000,
                 buffer_size=10000,
                 batch_size=64):
        self.policy_net = policy_net
        self.target_net = target_net
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.buffer = deque(maxlen=buffer_size)
        self.steps_done = 0

    def compute_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-1. * self.steps_done / self.epsilon_decay)

    @torch.no_grad()
    def select_action(self, state, deterministic=False):
        self.steps_done += 1
        epsilon = self.compute_epsilon()

        if not deterministic and random.random() < epsilon:
            return random.randint(0, self.policy_net(state).shape[-1] - 1)

        q_values = self.policy_net(state)

        if deterministic:
            return q_values.argmax(dim=-1).item()
        else:
            probs = torch.softmax(q_values, dim=-1)
            return torch.multinomial(probs, num_samples=1).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states, dim=0)
        next_states = torch.cat(next_states, dim=0)
        actions = torch.tensor(actions).unsqueeze(1)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        q_values = self.policy_net(states).gather(1, actions)
        next_q = self.target_net(next_states).max(dim=1, keepdim=True)[0]
        target = rewards + (1 - dones) * self.gamma * next_q

        loss = torch.nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
