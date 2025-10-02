"""
environment/sentiment_env.py

A lightweight RL environment wrapper around Keras model training for sentiment analysis.

Design:
- Each `step(action)` applies the requested hyperparameter changes (lr, batch_size, dropout),
  trains the model for `step_epochs` on the training split, evaluates on validation split,
  and returns (state, reward, done, info).

State = [val_accuracy, val_loss, epoch]
Reward = (val_accuracy - previous_val_accuracy) * reward_scaling  (positive if accuracy improved)
done = True when:
    - agent requests stop, OR
    - max_steps reached, OR
    - target_accuracy reached (optional)

Usage (example):
    env = SentimentEnv(X_train, y_train, X_val, y_val)
    state = env.reset()
    action = {"lr": 1e-3, "batch_size": 64}
    next_state, reward, done, info = env.step(action)
"""

import copy
import csv
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


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
        """
        Create environment.

        Params:
        - X_train, y_train, X_val, y_val: numpy arrays (preprocessed)
        - input_dim, maxlen, embedding_dim, lstm_units, dense_units: model architecture
        - initial_hyperparams: dict with keys 'lr', 'batch_size', 'dropout'
        - step_epochs: how many epochs to run per env.step()
        - max_steps: maximum steps per episode
        - reward_scaling: multiplier for accuracy delta -> reward
        - target_accuracy: if set, reaching it will end episode with positive reward
        - verbose: print training/eval info
        - random_seed: optional for reproducibility
        """
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
        # fill missing keys
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

        # build initial model
        self.model = self._build_model(self.current_hyperparams)
        # compiled inside _build_model

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
        """
        Reset the environment for a new episode.
        If reinit_model True, build a fresh model with current_hyperparams (weights reinitialized).
        Returns initial state (list of numeric values).
        """
        self.epoch = 0
        self.step_count = 0
        self.prev_val_acc = 0.0
        self.prev_val_loss = 1.0
        self.best_val_acc = 0.0

        if reinit_model:
            # rebuild to reset weights (use same current_hyperparams)
            self.model = self._build_model(self.current_hyperparams)

        return self._get_state()

    def _get_state(self):
        """Return numeric state vector for the agent: [val_accuracy, val_loss, epoch]."""
        return [float(self.prev_val_acc), float(self.prev_val_loss), float(self.epoch)]

    def step(self, action):
        """
        Apply an action and advance training by `step_epochs`. Action can be:
          - dict with keys 'lr', 'batch_size', 'dropout', 'stop' (all optional)
          - or an index / generic representation (user should translate into dict)
        Returns: (next_state, reward, done, info)
        """
        # Validate action type: expecting dict
        if action is None:
            action = {}

        if not isinstance(action, dict):
            raise ValueError("Action expected as dict: e.g. {'lr':1e-3, 'batch_size':64}.")

        # Apply hyperparameter changes
        rebuild_required = False
        old_weights = None

        # Dropout change -> rebuild model (safe to set_weights after, dropout layers don't have weights)
        if "dropout" in action:
            new_dropout = float(action["dropout"])
            if abs(new_dropout - self.current_hyperparams.get("dropout", 0.0)) > 1e-6:
                if self.verbose:
                    print(f"[ENV] Changing dropout: {self.current_hyperparams.get('dropout')} -> {new_dropout}")
                # save weights to restore (most layer shapes remain identical)
                try:
                    old_weights = self.model.get_weights()
                except Exception:
                    old_weights = None
                self.current_hyperparams["dropout"] = new_dropout
                rebuild_required = True

        # Learning rate change -> update optimizer lr in-place
        if "lr" in action:
            new_lr = float(action["lr"])
            # set keras optimizer lr variable safely
            try:
                K.set_value(self.model.optimizer.learning_rate, new_lr)
                self.current_hyperparams["lr"] = new_lr
                if self.verbose:
                    print(f"[ENV] Updated learning rate -> {new_lr:.6f}")
            except Exception:
                # if optimizer not set / unexpected, rebuild model
                if self.verbose:
                    print("[ENV] failed to set optimizer.lr in-place, will rebuild model with new lr")
                self.current_hyperparams["lr"] = new_lr
                rebuild_required = True

        # Batch size change -> stored, training will use this batch size
        if "batch_size" in action:
            new_bs = int(action["batch_size"])
            if new_bs <= 0:
                raise ValueError("batch_size must be positive int")
            if self.verbose:
                print(f"[ENV] Updated batch size -> {new_bs}")
            self.current_hyperparams["batch_size"] = new_bs

        # If rebuild required (dropout or forced), rebuild model and try to restore weights
        if rebuild_required:
            self.model = self._build_model(self.current_hyperparams)
            if old_weights is not None:
                try:
                    self.model.set_weights(old_weights)
                    if self.verbose:
                        print("[ENV] Restored previous weights after rebuild.")
                except Exception:
                    # If set_weights fails (rare if layer shapes changed), ignore and continue
                    if self.verbose:
                        print("[ENV] Could not restore weights after rebuild (shapes mismatch).")

        # Check for 'stop' action
        requested_stop = bool(action.get("stop", False))

        # Train for step_epochs
        bs = int(self.current_hyperparams.get("batch_size", self.default_hyperparams["batch_size"]))
        epochs_to_train = int(self.step_epochs)

        # guard: if nothing to train (empty dataset) raise
        if len(self.X_train) == 0:
            raise RuntimeError("Empty training dataset in environment.")

        # Fit quietly
        history = self.model.fit(
            self.X_train,
            self.y_train,
            epochs=epochs_to_train,
            batch_size=bs,
            validation_data=(self.X_val, self.y_val),
            verbose=0,
        )

        # update counters
        self.epoch += epochs_to_train
        self.step_count += 1

        # Evaluate on validation set
        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)

        # reward = improvement in val_acc compared to previous val_acc
        delta_acc = float(val_acc) - float(self.prev_val_acc)
        reward = float(delta_acc) * float(self.reward_scaling)

        # small penalty for many steps (optional)
        # reward -= 0.01 * epochs_to_train

        # update prev metrics
        self.prev_val_acc = float(val_acc)
        self.prev_val_loss = float(val_loss)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc

        # determine done
        done = False
        if requested_stop:
            done = True
            if self.verbose:
                print("[ENV] Stopping requested by agent.")
        if self.step_count >= self.max_steps:
            done = True
            if self.verbose:
                print("[ENV] Max steps reached.")
        if (self.target_accuracy is not None) and (val_acc >= self.target_accuracy):
            done = True
            # give a small bonus for reaching target
            reward += 5.0
            if self.verbose:
                print(f"[ENV] Target accuracy reached: {val_acc:.4f} >= {self.target_accuracy}")

        # Info dictionary (helpful for logs)
        info = {
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "epoch": int(self.epoch),
            "step_count": int(self.step_count),
            "hyperparams": copy.deepcopy(self.current_hyperparams),
            "best_val_accuracy": float(self.best_val_acc),
            "history_last": {k: v[-1] for k, v in history.history.items()} if hasattr(history, "history") else {},
        } 
        LOG_FILE = "results/accuracy_logs.csv"
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["step", "epoch", "val_accuracy", "reward", "learning_rate", "batch_size", "dropout"])

        if self.verbose:
            print(
                f"[ENV] Step {self.step_count} | epoch={self.epoch} | val_acc={val_acc:.4f} | reward={reward:.4f} | params={self.current_hyperparams}"
            )
 

        with open(LOG_FILE, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
            self.step_count,
            self.epoch,
            val_acc,
            reward,
            self.current_hyperparams['lr'],
            self.current_hyperparams['batch_size'],
            self.current_hyperparams['dropout'],
            
    ])

        return self._get_state(), float(reward), bool(done), info

    def render(self):
        """Print the current env status (for debugging)."""
        print(f"Epoch: {self.epoch}, Step: {self.step_count}, PrevValAcc: {self.prev_val_acc:.4f}, BestValAcc: {self.best_val_acc:.4f}")
        print("Current hyperparameters:", self.current_hyperparams)




# def step(self, action):
#     # ... your training code here ...
#     # values you already compute:
#     epoch = self.epoch
#     val_accuracy = val_acc
#     reward = reward
#     hyperparams = self.current_hyperparams  # or however you store params

#     # Logging to CSV
#     log_file = "results/accuracy_logs.csv"
#     if not os.path.exists(log_file):
#         with open(log_file, mode="w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow(["epoch", "val_accuracy", "reward", "learning_rate", "batch_size", "dropout"])

#     with open(log_file, mode="a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([
#             epoch,
#             val_accuracy,
#             reward,
#             hyperparams['lr'],
#             hyperparams['batch_size'],
#             hyperparams['dropout']
#         ])

#     # Finally return your usual state, reward, done, info
#     return state, reward, done, info
