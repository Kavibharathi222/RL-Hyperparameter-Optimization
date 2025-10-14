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
    prev_count =0
    epoch_count = 0

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

        # Training first to compute reward
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

        # Validation evaluation
        val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
        reward = (val_acc - self.prev_val_acc) * 50 + 0.1 * val_acc

        # ---------- Only update hyperparameters if reward is negative ----------
        if reward < 0:
            if "lr" in action:
                new_lr = float(action["lr"])
                self.current_hyperparams["lr"] = new_lr
                if hasattr(self.model.optimizer.learning_rate, "assign"):
                    self.model.optimizer.learning_rate.assign(new_lr)
                else:
                    self.model.optimizer.learning_rate = new_lr

            if "batch_size" in action:
                self.current_hyperparams["batch_size"] = int(action["batch_size"])

            if "dropout" in action:
                old_weights = self.model.get_weights()
                self.current_hyperparams["dropout"] = float(action["dropout"])
                self.model = self._build_model(self.current_hyperparams)
                try:
                    self.model.set_weights(old_weights)
                except Exception:
                    pass

        # Update prev_val_acc and prev_val_loss after step
        self.prev_val_acc = val_acc
        self.prev_val_loss = val_loss
        self.best_val_acc = max(self.best_val_acc, val_acc)

        done = self.step_count >= self.max_steps
        if self.target_accuracy and val_acc >= self.target_accuracy:
            reward += 5.0
            done = True

        # Logging (same as before)
        train_loss = history.history["loss"][-1]
        train_acc = history.history["accuracy"][-1]
        if SentimentEnv.prev_count ==0:
            SentimentEnv.prev_count=self.epoch
        else:
            SentimentEnv.prev_count+=1
        
        safe_append_csv(self.LOG_FILE, [
            self.step_count, SentimentEnv.prev_count, train_loss, train_acc,
            val_loss, val_acc, reward, self.best_val_acc,
            self.current_hyperparams["lr"],
            self.current_hyperparams["batch_size"],
            self.current_hyperparams["dropout"]
        ])
        try:
            shutil.copy(self.LOG_FILE, "results/accuracy_logs_latest.csv")
        except Exception:
            pass

        info = {
            "val_accuracy": val_acc,
            "val_loss": val_loss,
            "epoch": self.epoch,
            "step_count": self.step_count,
            "hyperparams": copy.deepcopy(self.current_hyperparams),
            "best_val_accuracy": self.best_val_acc,
        }

        if self.verbose:
            print(f"[ENV] Step {self.step_count} | Epoch={self.epoch} | "
                f"ValAcc={val_acc:.4f} | Reward={reward:.4f} | Params={self.current_hyperparams}")

        return self._get_state(), reward, done, info



    def render(self):
        print(f"Epoch: {self.epoch}, Step: {self.step_count}, "
              f"PrevValAcc: {self.prev_val_acc:.4f}, BestValAcc: {self.best_val_acc:.4f}")
        print("Current hyperparameters:", self.current_hyperparams)



# Updated code after charge 

# import copy
# import csv
# import os
# import time
# import datetime
# import shutil
# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import EarlyStopping, Callback

# # ---------- CSV Logging Helper ----------
# def safe_append_csv(log_file, row):
#     """Append safely to CSV (avoid Excel lock issues)."""
#     for _ in range(5):
#         try:
#             with open(log_file, mode="a", newline="") as f:
#                 writer = csv.writer(f)
#                 writer.writerow(row)
#             return
#         except PermissionError:
#             print(f"[WARN] File {log_file} busy. Retrying...")
#             time.sleep(1)

# # ---------- Custom Cyclical Learning Rate ----------
# class CyclicalLR(Callback):
#     """Cyclical learning rate scheduler for Keras."""
#     def __init__(self, base_lr=1e-4, max_lr=1e-2, step_size=5, scale_fn=None, scale_mode='cycle'):
#         super().__init__()
#         self.base_lr = base_lr
#         self.max_lr = max_lr
#         self.step_size = step_size
#         self.scale_fn = scale_fn or (lambda x: 1.)
#         self.scale_mode = scale_mode
#         self.iterations = 0
#         self.history = {}

#     def clr(self):
#         cycle = np.floor(1 + self.iterations / (2 * self.step_size))
#         x = np.abs(self.iterations / self.step_size - 2 * cycle + 1)
#         lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * self.scale_fn(
#             cycle if self.scale_mode == 'cycle' else self.iterations)
#         return lr

#     def on_train_begin(self, logs=None):
#         if hasattr(self.model.optimizer, 'lr'):
#             tf.keras.backend.set_value(self.model.optimizer.lr, self.base_lr)
#         elif hasattr(self.model.optimizer, 'learning_rate'):
#             tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.base_lr)
#         else:
#             raise ValueError('Optimizer must have a "lr" or "learning_rate" attribute.')

#     def on_batch_end(self, batch, logs=None):
#         self.iterations += 1
#         lr = self.clr()
#         if hasattr(self.model.optimizer, 'lr'):
#             tf.keras.backend.set_value(self.model.optimizer.lr, lr)
#         elif hasattr(self.model.optimizer, 'learning_rate'):
#             tf.keras.backend.set_value(self.model.optimizer.learning_rate, lr)
#         self.history.setdefault('lr', []).append(lr)

# # ---------- RL Environment ----------
# class SentimentEnv:
#     prev_count = 0

#     def __init__(self,
#                  X_train, y_train, X_val, y_val,
#                  input_dim=10000, maxlen=200,
#                  embedding_dim=128, lstm_units=64,
#                  dense_units=64, initial_hyperparams=None,
#                  step_epochs=1, max_steps=50,
#                  reward_scaling=100.0, target_accuracy=None,
#                  verbose=True, random_seed=None, new_run=True,
#                  plateau_patience=5, max_lr=1e-2):
#         if random_seed is not None:
#             np.random.seed(random_seed)
#             tf.random.set_seed(random_seed)

#         self.X_train, self.y_train = X_train, y_train
#         self.X_val, self.y_val = X_val, y_val
#         self.input_dim, self.maxlen = input_dim, maxlen
#         self.embedding_dim, self.lstm_units = embedding_dim, lstm_units
#         self.dense_units = dense_units

#         self.default_hyperparams = {"lr": 1e-3, "batch_size": 64, "dropout": 0.5}
#         self.current_hyperparams = {**self.default_hyperparams, **(initial_hyperparams or {})}

#         self.step_epochs = step_epochs
#         self.max_steps = max_steps
#         self.reward_scaling = reward_scaling
#         self.target_accuracy = target_accuracy
#         self.verbose = verbose
#         self.plateau_patience = plateau_patience
#         self.max_lr = max_lr

#         # Trackers
#         self.epoch = 0
#         self.step_count = 0
#         self.prev_val_acc = 0.0
#         self.prev_val_loss = 1.0
#         self.prev_train_acc = 0.0
#         self.best_val_acc = 0.0
#         self.prev_reward = 0.0
#         self.val_acc_history = []
#         self.final_model_path = "results/final_model.keras"

#         # Logging
#         os.makedirs("results", exist_ok=True)
#         timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") if new_run else ""
#         self.LOG_FILE = f"results/accuracy_logs_{timestamp}.csv" if new_run else "results/accuracy_logs_latest.csv"
#         with open(self.LOG_FILE, "w", newline="") as f:
#             writer = csv.writer(f)
#             writer.writerow([
#                 "step", "epoch", "train_loss", "train_accuracy",
#                 "val_loss", "val_accuracy", "reward", "best_val_accuracy",
#                 "learning_rate", "batch_size", "dropout"
#             ])

#         # Build initial model
#         self.model = self._build_model(self.current_hyperparams)

#     # ----------------- Model Builder -----------------
#     def _build_model(self, hyperparams):
#         reg = hyperparams.get("l2_reg", 0.001)
#         model = Sequential([
#             Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim),
#             Bidirectional(LSTM(self.lstm_units, return_sequences=False, kernel_regularizer=l2(reg))),
#             Dropout(float(hyperparams.get("dropout", 0.5))),
#             Dense(self.dense_units, activation="relu", kernel_regularizer=l2(reg)),
#             Dropout(float(hyperparams.get("dropout", 0.5))),
#             Dense(1, activation="sigmoid")
#         ])
#         opt = Adam(learning_rate=float(hyperparams.get("lr", 1e-3)))
#         model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
#         return model

#     # ----------------- Step Function -----------------
#     def step(self, action=None):
#         if action is None:
#             action = {}

#         # Adjust hyperparameters if last reward < 0
#         if self.prev_reward < 0:
#             # Learning rate
#             if "lr" in action:
#                 new_lr = min(float(action["lr"]), self.max_lr)
#                 if abs(new_lr - self.current_hyperparams["lr"]) > 1e-8:
#                     self.current_hyperparams["lr"] = new_lr
#                     tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
#                     if self.verbose: print(f"[ENV] Learning rate set to {new_lr:.6f}")
#             # Batch size
#             if "batch_size" in action:
#                 new_bs = int(action["batch_size"])
#                 if new_bs != self.current_hyperparams["batch_size"]:
#                     self.current_hyperparams["batch_size"] = new_bs
#                     if self.verbose: print(f"[ENV] Batch size set to {new_bs}")
#             # Dropout
#             if "dropout" in action:
#                 new_dp = float(action["dropout"])
#                 if abs(new_dp - self.current_hyperparams["dropout"]) > 1e-6:
#                     old_weights = self.model.get_weights()
#                     self.current_hyperparams["dropout"] = new_dp
#                     self.model = self._build_model(self.current_hyperparams)
#                     try: self.model.set_weights(old_weights)
#                     except: pass
#                     if self.verbose: print(f"[ENV] Dropout set to {new_dp:.2f}")

#         # Callbacks
#         early_stop = EarlyStopping(monitor='val_accuracy', patience=self.plateau_patience, restore_best_weights=True, verbose=0)
#         clr = CyclicalLR(base_lr=self.current_hyperparams["lr"]*0.5,
#                          max_lr=self.current_hyperparams["lr"],
#                          step_size=2)

#         # Train model
#         history = self.model.fit(
#             self.X_train, self.y_train,
#             epochs=self.step_epochs,
#             batch_size=self.current_hyperparams["batch_size"],
#             validation_data=(self.X_val, self.y_val),
#             verbose=0,
#             callbacks=[early_stop, clr]
#         )

#         self.epoch += self.step_epochs
#         self.step_count += 1

#         # Evaluate
#         val_loss, val_acc = self.model.evaluate(self.X_val, self.y_val, verbose=0)
#         train_loss, train_acc = history.history["loss"][-1], history.history["accuracy"][-1]

#         # Reward calculation
#         reward = (val_acc - self.prev_val_acc) * 50 + 0.1 * val_acc
#         reward -= max(0, train_acc - val_acc) * 10
#         self.prev_val_acc, self.prev_val_loss, self.prev_train_acc = val_acc, val_loss, train_acc
#         self.prev_reward = reward
#         self.best_val_acc = max(self.best_val_acc, val_acc)

#         # Track plateau
#         self.val_acc_history.append(val_acc)
#         if len(self.val_acc_history) > self.plateau_patience:
#             self.val_acc_history.pop(0)
#         plateau = len(self.val_acc_history) == self.plateau_patience and max(self.val_acc_history)-min(self.val_acc_history)<1e-4

#         done = self.step_count >= self.max_steps or plateau
#         if self.target_accuracy and val_acc >= self.target_accuracy:
#             done = True

#         # Logging
#         if SentimentEnv.prev_count == 0: SentimentEnv.prev_count = self.epoch
#         else: SentimentEnv.prev_count += 1
#         safe_append_csv(self.LOG_FILE, [
#             self.step_count, self.epoch, train_loss, train_acc,
#             val_loss, val_acc, reward, self.best_val_acc,
#             self.current_hyperparams["lr"],
#             self.current_hyperparams["batch_size"],
#             self.current_hyperparams["dropout"]
#         ])
#         try: shutil.copy(self.LOG_FILE, "results/accuracy_logs_latest.csv")
#         except: pass

#         if self.verbose:
#             print(f"[ENV] Step {self.step_count} | Epoch={self.epoch} | TrainAcc={train_acc:.4f} | ValAcc={val_acc:.4f} | Reward={reward:.4f}")

#         info = {
#             "val_accuracy": val_acc,
#             "val_loss": val_loss,
#             "train_accuracy": train_acc,
#             "epoch": self.epoch,
#             "step_count": self.step_count,
#             "hyperparams": copy.deepcopy(self.current_hyperparams),
#             "best_val_accuracy": self.best_val_acc
#         }

#         return self._get_state(), reward, done, info

#     # ----------------- Reset Environment -----------------
#     def reset(self, reinit_model=True):
#         self.epoch = 0
#         self.step_count = 0
#         self.prev_val_acc = 0.0
#         self.prev_val_loss = 1.0
#         self.prev_train_acc = 0.0
#         self.prev_reward = 0.0
#         self.val_acc_history = []
#         if reinit_model: self.model = self._build_model(self.current_hyperparams)
#         return self._get_state()

#     # ----------------- Get State -----------------
#     def _get_state(self):
#         return [float(self.prev_val_acc), float(self.prev_val_loss), float(self.epoch)]

#     # ----------------- Save Final Model -----------------
#     def save_final_model(self):
#         self.model.save(self.final_model_path)
#         print(f"[INFO] Final model saved at {self.final_model_path}")
