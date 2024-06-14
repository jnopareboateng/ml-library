# Qtable_RL.py

import numpy as np

class QTableRL:
    def __init__(self, state_size, action_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, max_exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay_rate=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.max_exploration_rate = max_exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay_rate = exploration_decay_rate
        self.q_table = np.zeros((state_size, action_size))

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.exploration_rate:
            return np.random.choice(self.action_size)
        else:
            return np.argmax(self.q_table[state, :])

    def update_q_table(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state, :])
        td_target = reward + self.discount_factor * self.q_table[next_state, best_next_action]
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        self.exploration_rate = self.min_exploration_rate + (self.max_exploration_rate - self.min_exploration_rate) * np.exp(-self.exploration_decay_rate)

    def train(self, episodes, env):
        rewards = []
        for episode in range(episodes):
            state = env.reset()
            total_rewards = 0
            done = False
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, info = env.step(action)
                self.update_q_table(state, action, reward, next_state)
                state = next_state
                total_rewards += reward
            rewards.append(total_rewards)
        return rewards

# Example usage
# env = SomeEnvironment()
# q_learning_agent = QTableRL(state_size=env.state_size, action_size=env.action_size)
# rewards = q_learning_agent.train(episodes=1000, env=env)
