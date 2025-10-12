import copy
import csv
import os
import time
import datetime
import shutil
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
    epoch_count = 0
    prev_epoch =0
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
        new_run=True,  # <-- new flag to control fresh file creation
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

        # ---------- Logging setup ----------
        os.makedirs("results", exist_ok=True)

        if new_run:
            # Create a new log file for each run with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.LOG_FILE = f"results/accuracy_logs_{timestamp}.csv"
            print(f"[INFO] New training session started. Log file: {self.LOG_FILE}")
        else:
            # Continue appending to latest file
            self.LOG_FILE = "results/accuracy_logs_latest.csv"
            print(f"[INFO] Continuing training. Appending to: {self.LOG_FILE}")

        # Write header for new run
        with open(self.LOG_FILE, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "step", "epoch", "train_loss", "train_accuracy",
                "val_loss", "val_accuracy", "reward", "best_val_accuracy",
                "learning_rate", "batch_size", "dropout"
            ])

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

        # Dynamic hyperparameter updates
        if "lr" in action:
            old_lr = self.current_hyperparams["lr"]
            new_lr = max(old_lr * float(action["lr"]), 1e-6)
            self.current_hyperparams["lr"] = new_lr
            if hasattr(self.model.optimizer.learning_rate, "assign"):
                self.model.optimizer.learning_rate.assign(new_lr)
            else:
                self.model.optimizer.learning_rate = new_lr
            if self.verbose:
                print(f"[ENV] LR adjusted: {old_lr:.6f} -> {new_lr:.6f}")

        if "batch_size" in action:
            old_bs = self.current_hyperparams["batch_size"]
            new_bs = int(min(max(old_bs * float(action["batch_size"]), 16), 256))
            self.current_hyperparams["batch_size"] = new_bs
            if self.verbose:
                print(f"[ENV] Batch size adjusted: {old_bs} -> {new_bs}")

        if "dropout" in action:
            old_dp = self.current_hyperparams["dropout"]
            new_dp = float(np.clip(old_dp * float(action["dropout"]), 0.1, 0.7))
            if abs(new_dp - old_dp) > 1e-4:
                old_weights = self.model.get_weights()
                self.current_hyperparams["dropout"] = new_dp
                self.model = self._build_model(self.current_hyperparams)
                try:
                    self.model.set_weights(old_weights)
                except Exception:
                    pass
                if self.verbose:
                    print(f"[ENV] Dropout adjusted: {old_dp:.2f} -> {new_dp:.2f}")

        requested_stop = bool(action.get("stop", False))

        # Train model
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

        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        delta_acc = val_acc - self.prev_val_acc
        reward = delta_acc * self.reward_scaling

        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss
        self.best_val_acc = max(self.best_val_acc, val_acc)

        done = requested_stop or self.step_count >= self.max_steps
        if self.target_accuracy and val_acc >= self.target_accuracy:
            reward += 5.0
            done = True

        # Save logs
        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]
        if SentimentEnv.prev_epoch==0:
            SentimentEnv.prev_epoch = self.epoch 
        else :
            SentimentEnv.prev_epoch +=1
        # SentimentEnv.epoch_count = self.step_count * 10 + self.epoch

        safe_append_csv(self.LOG_FILE, [
            self.step_count, SentimentEnv.prev_epoch, train_loss, train_acc,
            val_loss, val_acc, reward, self.best_val_acc,
            self.current_hyperparams["lr"],
            self.current_hyperparams["batch_size"],
            self.current_hyperparams["dropout"]
        ])

        # Copy this file as "latest" for easy access
        try:
            shutil.copy(self.LOG_FILE, "results/accuracy_logs_latest.csv")
        except Exception:
            pass

        if self.verbose:
            print(f"[ENV] Step {self.step_count} | Epoch={self.epoch} | "
                  f"ValAcc={val_acc:.4f} | Reward={reward:.4f} | Params={self.current_hyperparams}")

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
