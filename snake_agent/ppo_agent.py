import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class PPOAgent:
    def __init__(self, model, lr=1e-4, gamma=0.99, lam=0.95, clip_eps=0.2, entropy_coef=0.01):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.gamma = gamma
        self.lam = lam
        self.clip_eps = clip_eps
        self.entropy_coef = entropy_coef

    @torch.no_grad()
    def select_action(self, state):
        state_batch = state.unsqueeze(0)
        logits, value = self.model(state_batch)
        dist = Categorical(logits=logits)
        action = dist.sample()

        return (
            action.item(),
            dist.log_prob(action).item(),
            value.item()
        )

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        returns = [a + v for a, v in zip(advantages, values[:-1])]

        return advantages, returns

    def update(self, rollouts, epochs=4, batch_size=64):
        states, actions, log_probs_old, returns, advantages = rollouts

        device = next(self.model.parameters()).device

        max_H = max(s.shape[1] for s in states)
        max_W = max(s.shape[2] for s in states)

        padded_states = []
        for s in states:
            C, H, W = s.shape
            pad_h = max_H - H
            pad_w = max_W - W
            s_padded = F.pad(s, (0, pad_w, 0, pad_h))
            padded_states.append(s_padded)

        states = torch.stack(padded_states).to(device)

        actions = torch.tensor(actions).to(device)
        log_probs_old = torch.tensor(log_probs_old).to(device)
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = torch.tensor(advantages, dtype=torch.float32).to(device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset = torch.utils.data.TensorDataset(states, actions, log_probs_old, returns, advantages)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for _ in range(epochs):
            for s, a, old_logp, ret, adv in loader:
                s = s.to(device)
                a = a.to(device)
                old_logp = old_logp.to(device)
                ret = ret.to(device)
                adv = adv.to(device)

                logits, value = self.model(s)
                dist = Categorical(logits=logits)
                logp = dist.log_prob(a)
                entropy = dist.entropy().mean()

                ratio = torch.exp(logp - old_logp)
                surrogate1 = ratio * adv
                surrogate2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * adv
                policy_loss = -torch.min(surrogate1, surrogate2).mean()
                value_loss = F.mse_loss(value, ret)
                loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
