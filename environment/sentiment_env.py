import copy
import csv
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


# ---------- CSV Logging Helper ----------
def safe_append_csv(log_file, row):
    """Append safely to CSV (avoid Excel lock issues)."""
    for _ in range(5):
        try:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return
        except PermissionError:
            print(f"[WARN] File {log_file} busy. Retrying...")
            time.sleep(1)


# ---------- RL Environment ----------
class SentimentEnv:
    def __init__(
        self,
        X_train, y_train,
        X_val, y_val,
        input_dim=10000,
        maxlen=200,
        embedding_dim=128,
        lstm_units=64,
        dense_units=64,
        initial_hyperparams=None,
        step_epochs=1,
        max_steps=50,
        reward_scaling=100.0,
        target_accuracy=None,
        verbose=True,
        random_seed=None,
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.input_dim = input_dim
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        # Initial hyperparameters
        self.default_hyperparams = {"lr": 1e-3, "batch_size": 64, "dropout": 0.5}
        self.current_hyperparams = {**self.default_hyperparams, **(initial_hyperparams or {})}

        self.step_epochs = step_epochs
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling
        self.target_accuracy = target_accuracy
        self.verbose = verbose

        # Internal trackers
        self.epoch = 0
        self.step_count = 0
        self.prev_val_acc = 0.0
        self.prev_val_loss = 1.0
        self.best_val_acc = 0.0

        os.makedirs("results", exist_ok=True)
        self.LOG_FILE = "results/accuracy_logs.csv"

        # Build initial model
        self.model = self._build_model(self.current_hyperparams)

    # ---------- Model Builder ----------
    def _build_model(self, hyperparams):
        model = Sequential([
            Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim),
            Bidirectional(LSTM(self.lstm_units, return_sequences=False)),
            Dropout(float(hyperparams.get("dropout", 0.5))),
            Dense(self.dense_units, activation="relu"),
            Dropout(float(hyperparams.get("dropout", 0.5))),
            Dense(1, activation="sigmoid"),
        ])
        opt = Adam(learning_rate=float(hyperparams.get("lr", 1e-3)))
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return model

    # ---------- Reset ----------
    def reset(self, reinit_model=True):
        self.epoch = 0
        self.step_count = 0
        self.prev_val_acc = 0.0
        self.prev_val_loss = 1.0
        self.best_val_acc = 0.0
        if reinit_model:
            self.model = self._build_model(self.current_hyperparams)
        return self._get_state()

    def _get_state(self):
        return [float(self.prev_val_acc), float(self.prev_val_loss), float(self.epoch)]

    # ---------- Step ----------
    def step(self, action=None):
        if action is None:
            action = {}

        # --- Train model ---
        bs = self.current_hyperparams["batch_size"]
        history = self.model.fit(
            self.X_train, self.y_train,
            epochs=self.step_epochs,
            batch_size=bs,
            validation_data=(self.X_val, self.y_val),
            verbose=0,
        )

        self.epoch += self.step_epochs
        self.step_count += 1

        # Evaluate
        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        delta_acc = val_acc - self.prev_val_acc
        reward = delta_acc * self.reward_scaling

        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss
        self.best_val_acc = max(self.best_val_acc, val_acc)
        print("Learning Rate ",self.current_hyperparams["lr"])
        print("Droup Out ",self.current_hyperparams["dropout"])
        print("Droupout ", self.current_hyperparams["batch_size"])
        # --- 🔑 Adjust hyperparameters if reward is negative ---
        if reward < 0:
            # Reduce LR a bit
            old_lr = self.current_hyperparams["lr"]
            new_lr = max(old_lr * 0.8, 1e-6)
            self.current_hyperparams["lr"] = new_lr
            if hasattr(self.model.optimizer.learning_rate, "assign"):
                self.model.optimizer.learning_rate.assign(new_lr)
            else:
                self.model.optimizer.learning_rate = new_lr
            if self.verbose:
                print(f"[ENV] Reward < 0 → LR reduced {old_lr:.6f} → {new_lr:.6f}")

            # Adjust dropout slightly
            old_dp = self.current_hyperparams["dropout"]
            new_dp = float(np.clip(old_dp * 1.1, 0.1, 0.7))
            self.current_hyperparams["dropout"] = new_dp
            if abs(new_dp - old_dp) > 1e-4:
                old_weights = self.model.get_weights()
                self.model = self._build_model(self.current_hyperparams)
                try:
                    self.model.set_weights(old_weights)
                except Exception:
                    pass
            if self.verbose:
                print(f"[ENV] Reward < 0 → Dropout adjusted {old_dp:.2f} → {new_dp:.2f}")

        # --- Check stopping ---
        done = self.step_count >= self.max_steps
        if self.target_accuracy and val_acc >= self.target_accuracy:
            reward += 5.0
            done = True

        # --- Logging ---
        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]

        safe_append_csv(self.LOG_FILE, [
            self.step_count, self.epoch, train_loss, train_acc,
            val_loss, val_acc, reward, self.best_val_acc,
            self.current_hyperparams["lr"],
            self.current_hyperparams["batch_size"],
            self.current_hyperparams["dropout"]
        ])

        info = {
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "epoch": self.epoch,
            "step_count": self.step_count,
            "hyperparams": copy.deepcopy(self.current_hyperparams),
            "best_val_accuracy": self.best_val_acc,
        }

        return self._get_state(), reward, done, info


    def render(self):
        print(f"Epoch: {self.epoch}, Step: {self.step_count}, "
              f"PrevValAcc: {self.prev_val_acc:.4f}, BestValAcc: {self.best_val_acc:.4f}")
        print("Current hyperparameters:", self.current_hyperparams)