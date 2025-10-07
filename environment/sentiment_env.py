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
    for _ in range(5):
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

        # log file
        self.LOG_FILE = "results/accuracy_logs.csv"

        # build initial model
        self.model = self._build_model(self.current_hyperparams)

    def _build_model(self, hyperparams):
        model = Sequential()
        model.add(Embedding(input_dim=self.input_dim, output_dim=self.embedding_dim))
        model.add(Bidirectional(LSTM(self.lstm_units, return_sequences=False)))
        model.add(Dropout(float(hyperparams.get("dropout", 0.5))))
        model.add(Dense(self.dense_units, activation="relu"))
        model.add(Dropout(float(hyperparams.get("dropout", 0.5))))
        model.add(Dense(1, activation="sigmoid"))

        opt = Adam(learning_rate=float(hyperparams.get("lr", 1e-3)))
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

    def step(self, action=None):
        # ✅ Enforce correct types
        self.current_hyperparams["lr"] = float(self.current_hyperparams.get("lr", 1e-3))
        self.current_hyperparams["batch_size"] = int(self.current_hyperparams.get("batch_size", 64))
        self.current_hyperparams["dropout"] = float(self.current_hyperparams.get("dropout", 0.5))

        if action is None:
            action = {}

        # update hyperparams safely
        if "lr" in action:
            self.current_hyperparams["lr"] = float(action["lr"])
            new_lr = float(self.current_hyperparams["lr"])
            if hasattr(self.model.optimizer.learning_rate, "assign"):
                self.model.optimizer.learning_rate.assign(new_lr)
            else:
                self.model.optimizer.learning_rate = new_lr


    # ✅ Safe update for both tf.Variable and float cases
    # if hasattr(self.model.optimizer.learning_rate, "assign"):
    #     self.model.optimizer.learning_rate.assign(new_lr)
    # # else:
    #     self.model.optimizer.learning_rate = new_lr


        if "batch_size" in action:
            self.current_hyperparams["batch_size"] = int(action["batch_size"])

        if "dropout" in action:
            self.current_hyperparams["dropout"] = float(action["dropout"])
            self.model = self._build_model(self.current_hyperparams)

        requested_stop = bool(action.get("stop", False))

        bs = int(self.current_hyperparams["batch_size"])
        epochs_to_train = int(self.step_epochs)

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

        # logging
        if not os.path.exists(self.LOG_FILE):
            with open(self.LOG_FILE, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "step", "epoch", "train_loss", "train_accuracy",
                    "val_loss", "val_accuracy", "reward", "best_val_accuracy",
                    "learning_rate", "batch_size", "dropout"
                ])

        train_loss = history.history["loss"][-1] if "loss" in history.history else None
        train_acc = history.history["accuracy"][-1] if "accuracy" in history.history else None

        safe_append_csv(self.LOG_FILE, [
            self.step_count, self.epoch, train_loss, train_acc,
            val_loss, val_acc, reward, self.best_val_acc,
            float(self.current_hyperparams['lr']),
            int(self.current_hyperparams['batch_size']),
            float(self.current_hyperparams['dropout'])
        ])

        info = {
            "val_accuracy": float(val_acc),
            "val_loss": float(val_loss),
            "epoch": int(self.epoch),
            "step_count": int(self.step_count),
            "hyperparams": copy.deepcopy(self.current_hyperparams),
            "best_val_accuracy": float(self.best_val_acc),
        }
        return self._get_state(), float(reward), bool(done), info

    def render(self):
        print(f"Epoch: {self.epoch}, Step: {self.step_count}, PrevValAcc: {self.prev_val_acc:.4f}, BestValAcc: {self.best_val_acc:.4f}")
        print("Current hyperparameters:", self.current_hyperparams)





























import os
import pickle
import matplotlib.pyplot as plt

from Preprocessing.feature_extraction import load_and_preprocess_imdb
from environment.sentiment_env import SentimentEnv
from models.rl_agent import DQNAgent

# -----------------------------
# 1. Load and preprocess IMDB dataset
# -----------------------------
print("[INFO] Loading and preprocessing dataset...")
X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# Validation split
X_val, y_val = X_train[:5000], y_train[:5000]
X_train, y_train = X_train[5000:], y_train[5000:]

# Save tokenizer for future inference
os.makedirs("models", exist_ok=True)
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to models/tokenizer.pkl")

# -----------------------------
# 2. Define action space for DQN
# -----------------------------
learning_rates = [0.001, 0.0005, 0.0001]
batch_sizes = [32, 64]
dropouts = [0.3, 0.5, 0.6]

action_space = [
    {"lr": lr, "batch_size": bs, "dropout": dr}
    for lr in learning_rates
    for bs in batch_sizes
    for dr in dropouts
]
print(f"[INFO] Action space size: {len(action_space)}")

# -----------------------------
# 3. Initialize environment and DQN agent
# -----------------------------
env = SentimentEnv(
    X_train, y_train, X_val, y_val,
    input_dim=10000,
    maxlen=200,
    step_epochs=1,
    max_steps=10,
    target_accuracy=0.87,   # RL steps (you can adjust)
    verbose=True
)

state_size = 3      # [val_accuracy, val_loss, epoch]
action_size = len(action_space)

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=0.001,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9,
    batch_size=16,
    memory_size=1000,
    target_update_freq=5
)

# -----------------------------
# 4. Training Loop (with Model Save)
# -----------------------------
episodes = 5
rewards_log = []
best_val_acc = 0.0
best_model_path = "models/best_model.h5"
final_model_path = "models/final_model.h5"

for ep in range(episodes):
    print(f"\n🚀 [EPISODE {ep + 1}/{episodes}] ----------------------------")
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # Choose action
        action_idx = agent.act(state)
        action = action_space[action_idx]

        # Environment step
        next_state, reward, done, info = env.step(action)

        # Remember experience
        agent.remember(state, action_idx, reward, next_state, done)

        # Train from replay buffer
        agent.replay()

        # Update totals
        state = next_state
        total_reward += reward

        # Save best model dynamically
        if info["val_accuracy"] > best_val_acc:
            best_val_acc = info["val_accuracy"]
            env.model.save(best_model_path)
            print(f"💾 [SAVE] New best model (val_acc={best_val_acc:.4f})")

        # If training stopped early
        if done:
            print("🛑 Training stopped (done=True). Saving current model...")
            env.model.save(final_model_path)
            break
    if done :
        break
      
        

    print(f"[EPISODE {ep + 1}] Total Reward = {total_reward:.2f} | Best Val Accuracy = {best_val_acc:.4f}")
    rewards_log.append(total_reward)

print("\n✅ Training completed successfully!")
print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")

# Save final model again at end
env.model.save(final_model_path)
print(f"💾 Final model saved to {final_model_path}")

# -----------------------------
# 5. Plot rewards per episode
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(rewards_log) + 1), rewards_log, marker="o", color="blue")
plt.title("DQN Training Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()




# import os
# import pickle
# import matplotlib.pyplot as plt
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from Preprocessing.feature_extraction import load_and_preprocess_imdb
# from environment.sentiment_env import SentimentEnv

# # ===============================
# # 1️⃣ Load and Preprocess Dataset
# # ===============================
# print("🔹 Loading and preprocessing IMDB dataset...")

# # Ensure your preprocessing returns tokenizer
# X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# # ===============================
# # 2️⃣ Initialize Environment
# # ===============================
# print("🔹 Initializing RL Environment...")
# env = SentimentEnv(
#     X_tr, y_tr,
#     X_val, y_val,
#     step_epochs=1,          # train 3 epochs per step
#     max_steps=10,          # up to 300 steps
#     target_accuracy=0.87,   # stop if 90% validation accuracy
#     verbose=True
# )

# state = env.reset()

# previous_acc = 0.0
# plateau_counter = 0

# # ===============================
# # 3️⃣ Adaptive RL Loop
# # ===============================
# print("🔹 Starting RL Agent training loop...")

# for step in range(env.max_steps):
#     prev_val_acc = float(state[0])
#     prev_val_loss = float(state[1])
#     minstep=6
#     action = {}

#     # === CONDITIONS ===

#     # Condition 1: Validation loss ↑ → reduce LR
#     if prev_val_loss > env.prev_val_loss:
#         action["lr"] = float(max(float(env.current_hyperparams["lr"]) * 0.5, 1e-6))

#     # Condition 3: Overfitting → increase dropout
#     if (env.prev_val_acc < env.best_val_acc) and (env.prev_val_loss < 0.5):
#         action["dropout"] = float(min(float(env.current_hyperparams["dropout"]) + 0.1, 0.7))

#     # Condition 4: Underfitting → decrease dropout or increase LR
#     if prev_val_acc < 0.65 and prev_val_loss > 0.7:
#         action["dropout"] = float(max(float(env.current_hyperparams["dropout"]) - 0.1, 0.1))
#         action["lr"] = float(min(float(env.current_hyperparams["lr"]) * 1.2, 1e-2))

#     # Condition 5: Plateau in accuracy → adjust
#     if abs(prev_val_acc - previous_acc) < 0.001:
#         plateau_counter += 1
#     else:
#         plateau_counter = 0
#     if plateau_counter >= 3:
#         action["lr"] = float(env.current_hyperparams["lr"]) * 0.7
#         action["batch_size"] = int(min(env.current_hyperparams["batch_size"] * 2, 256))
#         plateau_counter = 0

#     # Condition 6: Scheduled adjustment every 50 epochs
#     if env.epoch > 0 and env.epoch % 50 == 0:
#         action["lr"] = float(max(float(env.current_hyperparams["lr"]) * 0.8, 1e-6))

#     # Default: keep batch size same
#     action.setdefault("batch_size", int(env.current_hyperparams["batch_size"]))

#     # === Step ===
#     state, reward, done, info = env.step(action)
#     previous_acc = float(info["val_accuracy"])

#     print(f"Step {step+1} | Epoch {info['epoch']} | "
#           f"ValAcc: {info['val_accuracy']:.4f} | ValLoss: {info['val_loss']:.4f} | "
#           f"Reward: {reward:.4f} | Params: {info['hyperparams']}")

#     if done and step > minstep :
#         print("✅ Training stopped (target accuracy or max steps reached).")
#         break

# print("✅ RL Training completed successfully!")

# # ===============================
# # 4️⃣ Save Model + Tokenizer
# # ===============================
# os.makedirs("results", exist_ok=True)

# # Save model as Pickle
# model_pickle_path = "results/rl_trained_model.pkl"
# with open(model_pickle_path, "wb") as f:
#     pickle.dump(env.model, f)
# print(f"✅ Model saved (pickle) at: {model_pickle_path}")

# # Save model as Keras H5
# model_h5_path = "results/rl_trained_model.h5"
# env.model.save(model_h5_path)
# print(f"✅ Model saved (H5) at: {model_h5_path}")

# # Save tokenizer
# tokenizer_path = "results/tokenizer.pkl"
# with open(tokenizer_path, "wb") as f:
#     pickle.dump(tokenizer, f)
# print(f"✅ Tokenizer saved at: {tokenizer_path}")

# # ===============================
# # 5️⃣ Plot Training Logs
# # ===============================
# csv_path = "results/accuracy_logs.csv"
# if os.path.exists(csv_path):
#     logs = pd.read_csv(csv_path)

#     plt.figure(figsize=(8, 5))
#     plt.plot(logs['epoch'], logs['val_accuracy'], label="Validation Accuracy")
#     plt.plot(logs['epoch'], logs['reward'], label="Reward")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.title("RL Hyperparameter Optimization Progress")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
# else:
#     print("⚠️ Log file not found. Please ensure training completed successfully.")
