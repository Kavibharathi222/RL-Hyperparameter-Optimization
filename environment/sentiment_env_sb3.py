import gym
from gym import spaces
import numpy as np
import os
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense
from tensorflow.keras.optimizers import Adam

class SentimentEnvSB3(gym.Env):
    """
    SB3-compatible environment for hyperparameter optimization
    """
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        X_train, y_train, X_val, y_val,
        action_space_list,
        step_epochs=1,
        max_steps=10,
        target_accuracy=0.99,
        verbose=True
    ):
        super().__init__()

        self.X_train, self.y_train = X_train, y_train
        self.X_val, self.y_val = X_val, y_val
        self.action_space_list = action_space_list

        self.step_epochs = step_epochs
        self.max_steps = max_steps
        self.target_accuracy = target_accuracy
        self.verbose = verbose
        self.best_val_acc = 0.0
        self.best_hparams = None


        # ---- Gym spaces ----
        self.action_space = spaces.Discrete(len(action_space_list))
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

        self.reset()

    # ------------------------
    def _build_model(self, lr, dropout):
        model = Sequential([
            Embedding(10000, 128, input_length=200, trainable=False),
            Bidirectional(LSTM(200, dropout=dropout)),
            Dense(1, activation="sigmoid")
        ])
        model.compile(
            optimizer=Adam(learning_rate=lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    # ------------------------
    def reset(self):
        self.step_count = 0
        self.prev_val_acc = 0.0
        self.prev_val_loss = 1.0
        # self.step_count = 0
        # self.prev_val_acc = 0.0
        # self.prev_val_loss = 1.0


        default = self.action_space_list[0]
        self.model = self._build_model(default["lr"], default["dropout"])

        return np.array(
            [self.prev_val_acc, self.prev_val_loss, 0.0],
            dtype=np.float32
        )

    # ------------------------
    def step(self, action_idx):
        action = self.action_space_list[action_idx]

        # Update model hyperparameters
        self.model = self._build_model(
            action["lr"],
            action["dropout"]
        )

        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.step_epochs,
            batch_size=action["batch_size"],
            validation_data=(self.X_val, self.y_val),
            verbose=0
        )

        val_loss, val_acc = self.model.evaluate(
            self.X_val, self.y_val, verbose=0
        )

        reward = (val_acc - self.prev_val_acc) * 50
        reward -= max(0, history.history["accuracy"][-1] - val_acc) * 10

        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss
        self.step_count += 1

        done = (
            self.step_count >= self.max_steps
            or val_acc >= self.target_accuracy
        )

        # ------------------------
# Save best hyperparameters
# ------------------------
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_hparams = action

            os.makedirs("SavedModels", exist_ok=True)
            with open("SavedModels/best_hparams.pkl", "wb") as f:
                pickle.dump(self.best_hparams, f)

            if self.verbose:
                print(f"[ENV] ✅ Best hyperparams saved: {self.best_hparams}")


        obs = np.array(
            [val_acc, val_loss, self.step_count],
            dtype=np.float32
        )

        info = {
            "val_accuracy": val_acc,
            "hyperparams": action
        }

        if self.verbose:
            print(f"[ENV] Step={self.step_count} | ValAcc={val_acc:.4f} | Reward={reward:.4f}")

        return obs, reward, done, info
