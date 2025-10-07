# import random
# import numpy as np
# from collections import defaultdict

# class RLAgent:
#     def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.2):
#         """
#         RL Agent using Q-learning
#         :param actions: list of possible actions (hyperparameters)
#         :param alpha: learning rate for Q-update
#         :param gamma: discount factor
#         :param epsilon: exploration rate
#         """
#         self.q_table = defaultdict(float)
#         self.actions = actions
#         self.alpha = alpha
#         self.gamma = gamma
#         self.epsilon = epsilon

#     def choose_action(self, state):
#         """Epsilon-greedy action selection"""
#         if random.uniform(0, 1) < self.epsilon:
#             return random.choice(self.actions)  # Explore
#         else:
#             q_values = [self.q_table[(tuple(state), a)] for a in self.actions]
#             return self.actions[np.argmax(q_values)]  # Exploit

#     def update(self, state, action, reward, next_state):
#         """Q-learning update"""
#         best_next_action = max(
#             [self.q_table[(tuple(next_state), a)] for a in self.actions]
#         )
#         current_q = self.q_table[(tuple(state), action)]
#         self.q_table[(tuple(state), action)] = current_q + self.alpha * (
#             reward + self.gamma * best_next_action - current_q
#         )

import random
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam

class DQNAgent:
    def __init__(
        self,
        state_size,
        action_size,
        lr=0.001,
        gamma=0.95,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        batch_size=32,
        memory_size=2000,
        target_update_freq=10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        # Build main and target networks
        self.model = self._build_model(lr)
        self.target_model = self._build_model(lr)
        self.update_target_model()  # initialize target weights

        self.train_steps = 0

    def _build_model(self, lr):
        """Neural network approximating Q(s, a)."""
        model = Sequential([
            Input(shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(64, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=lr), loss='mse')
        return model

    def update_target_model(self):
        """Copy weights from main model to target model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience for replay."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Epsilon-greedy action selection."""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(np.array([state]), verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Train the model using random batch from memory."""
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, targets = [], []

        for state, action, reward, next_state, done in batch:
            target = self.model.predict(np.array([state]), verbose=0)[0]
            if done:
                target[action] = reward
            else:
                t = self.target_model.predict(np.array([next_state]), verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            states.append(state)
            targets.append(target)

        self.model.fit(np.array(states), np.array(targets), epochs=1, verbose=0)

        # Epsilon decay
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.train_steps += 1
        if self.train_steps % self.target_update_freq == 0:
            self.update_target_model()
