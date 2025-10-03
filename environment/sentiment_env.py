"""
environment/sentiment_env.py

A lightweight RL environment wrapper around Keras model training for sentiment analysis.
"""

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


def safe_append_csv(log_file, row):
    """Safe append to CSV (handles file lock issues)."""
    for _ in range(2):
        try:
            with open(log_file, mode="a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            return
        except PermissionError:
            print(f"[WARN] File {log_file} is busy (maybe open in Excel). Retrying...")
            time.sleep(1)


class SentimentEnv:
    def __init__(
        self,
        X_train,
        y_train,
        X_val,
        y_val,
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
        verbose=False,
        random_seed=None,
    ):
        if random_seed is not None:
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)

        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        # architecture params
        self.input_dim = input_dim
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.dense_units = dense_units

        # hyperparams
        self.default_hyperparams = {"lr": 1e-3, "batch_size": 64, "dropout": 0.5}
        if initial_hyperparams is None:
            initial_hyperparams = {}
        self.current_hyperparams = {**self.default_hyperparams, **initial_hyperparams}

        self.step_epochs = step_epochs
        self.max_steps = max_steps
        self.reward_scaling = reward_scaling
        self.target_accuracy = target_accuracy
        self.verbose = verbose

        # internal state
        self.epoch = 0
        self.step_count = 0
        self.prev_val_acc = 0.0
        self.prev_val_loss = 1.0
        self.best_val_acc = 0.0

        # CSV logging setup
        self.LOG_FILE = "results/accuracy_logs.csv"
        os.makedirs(os.path.dirname(self.LOG_FILE), exist_ok=True)
        if not os.path.exists(self.LOG_FILE) or os.stat(self.LOG_FILE).st_size == 0:
            with open(self.LOG_FILE, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "epoch",
                    "train_loss", "train_accuracy",
                    "val_loss", "val_accuracy",
                    "reward", "best_val_accuracy",
                    "learning_rate", "batch_size", "dropout"
                ])

        # build initial model
        self.model = self._build_model(self.current_hyperparams)

    def _build_model(self, hyperparams):
        """Builds and compiles a BiLSTM model using the provided hyperparams."""
        model = Sequential()
        model.add(Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim, input_length=self.maxlen))
        model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=False)))
        model.add(Dropout(hyperparams.get("dropout", self.default_hyperparams["dropout"])))
        model.add(Dense(self.dense_units, activation="relu"))
        model.add(Dropout(hyperparams.get("dropout", self.default_hyperparams["dropout"])))
        model.add(Dense(1, activation="sigmoid"))

        opt = Adam(learning_rate=hyperparams.get("lr", self.default_hyperparams["lr"]))
        model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
        return model

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

    def step(self, action):
        if action is None:
            action = {}
        if not isinstance(action, dict):
            raise ValueError("Action expected as dict: e.g. {'lr':1e-3, 'batch_size':64}.")

        # Handle hyperparam changes
        rebuild_required = False
        old_weights = None

        if "dropout" in action:
            new_dropout = float(action["dropout"])
            if abs(new_dropout - self.current_hyperparams.get("dropout", 0.0)) > 1e-6:
                if self.verbose:
                    print(f"[ENV] Changing dropout: {self.current_hyperparams.get('dropout')} -> {new_dropout}")
                try:
                    old_weights = self.model.get_weights()
                except Exception:
                    old_weights = None
                self.current_hyperparams["dropout"] = new_dropout
                rebuild_required = True

        if "lr" in action:
            new_lr = float(action["lr"])
            try:
                K.set_value(self.model.optimizer.learning_rate, new_lr)
                self.current_hyperparams["lr"] = new_lr
            except Exception:
                self.current_hyperparams["lr"] = new_lr
                rebuild_required = True

        if "batch_size" in action:
            new_bs = int(action["batch_size"])
            if new_bs <= 0:
                raise ValueError("batch_size must be positive int")
            self.current_hyperparams["batch_size"] = new_bs

        if rebuild_required:
            self.model = self._build_model(self.current_hyperparams)
            if old_weights is not None:
                try:
                    self.model.set_weights(old_weights)
                except Exception:
                    pass

        requested_stop = bool(action.get("stop", False))

        bs = int(self.current_hyperparams.get("batch_size", self.default_hyperparams["batch_size"]))
        epochs_to_train = int(self.step_epochs)

        if len(self.X_train) == 0:
            raise RuntimeError("Empty training dataset in environment.")

        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs_to_train,
            batch_size=bs,
            validation_data=(self.X_val, self.y_val),
            verbose=0,
        )

        self.epoch += epochs_to_train
        self.step_count += 1

        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)

        delta_acc = float(val_acc) - float(self.prev_val_acc)
        reward = float(delta_acc) * float(self.reward_scaling)

        self.prev_val_acc = float(val_acc)
        self.prev_val_loss = float(val_loss)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        done = False
        if requested_stop or self.step_count >= self.max_steps:
            done = True
        if (self.target_accuracy is not None) and (val_acc >= self.target_accuracy):
            done = True
            reward += 5.0

        info = {
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "epoch": int(self.epoch),
            "step_count": int(self.step_count),
            "hyperparams": copy.deepcopy(self.current_hyperparams),
            "best_val_accuracy": float(self.best_val_acc),
            "history_last": {k: v[-1] for k, v in history.history.items()} if hasattr(history, "history") else {},
        }

        # log row
        train_loss = history.history["loss"][-1] if "loss" in history.history else None
        train_acc = history.history["accuracy"][-1] if "accuracy" in history.history else None

        safe_append_csv(self.LOG_FILE, [
            self.step_count,
            self.epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            reward,
            self.best_val_acc,
            self.current_hyperparams['lr'],
            self.current_hyperparams['batch_size'],
            self.current_hyperparams['dropout']
        ])

        return self._get_state(), float(reward), bool(done), info

    def render(self):
        print(f"Epoch: {self.epoch}, Step: {self.step_count}, PrevValAcc: {self.prev_val_acc:.4f}, BestValAcc: {self.best_val_acc:.4f}")
        print("Current hyperparameters:", self.current_hyperparams)
