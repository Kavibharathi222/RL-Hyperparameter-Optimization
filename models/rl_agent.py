import random
import numpy as np
from collections import defaultdict

class RLAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
        """
        RL Agent using Q-learning
        :param actions: list of possible actions (hyperparameters)
        :param alpha: learning rate for Q-update
        :param gamma: discount factor
        :param epsilon: exploration rate
        """
        self.q_table = defaultdict(float)
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

    def choose_action(self, state):
        """Epsilon-greedy action selection"""
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  # Explore
        else:
            q_values = [self.q_table[(tuple(state), a)] for a in self.actions]
            return self.actions[np.argmax(q_values)]  # Exploit

    def update(self, state, action, reward, next_state):
        """Q-learning update"""
        best_next_action = max(
            [self.q_table[(tuple(next_state), a)] for a in self.actions]
        )
        current_q = self.q_table[(tuple(state), action)]
        self.q_table[(tuple(state), action)] = current_q + self.alpha * (
            reward + self.gamma * best_next_action - current_q
        )
