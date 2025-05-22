class RolloutBuffer:
    def __init__(self):
        self.clear()

    def add(self, state, action, log_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        return (
            self.states,
            self.actions,
            self.log_probs,
            self.rewards,
            self.values,
            self.dones,
        )

    def clear(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []