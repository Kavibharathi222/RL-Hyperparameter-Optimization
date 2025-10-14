import os
import pickle
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Preprocessing.feature_extraction import load_and_preprocess_imdb
from environment.sentiment_env import SentimentEnv
from models.rl_agent import DQNAgent
from utils.plot_utils import plot_training_logs

# -----------------------------
# 1️⃣ Load & Preprocess Dataset
# -----------------------------
print("[INFO] Loading and preprocessing IMDB dataset...")
X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# Split for validation
X_val, y_val = X_train[:5000], y_train[:5000]
X_train, y_train = X_train[5000:], y_train[5000:]

os.makedirs("models", exist_ok=True)
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to models/tokenizer.pkl")

# -----------------------------
# 2️⃣ Define Action Space
# -----------------------------
lr_values = [0.001, 0.0008, 0.0005]
batch_sizes = [64, 128, 256]
dropouts = [0.5]

action_space = [
    {"lr": lr, "batch_size": bs, "dropout": dr}
    for lr in lr_values
    for bs in batch_sizes
    for dr in dropouts
]
print(f"[INFO] Action space size: {len(action_space)}")

# -----------------------------
# 3️⃣ Initialize Environment + Agent
# -----------------------------
env = SentimentEnv(
    X_train, y_train, X_val, y_val,
    step_epochs=3,
    max_steps=10,
    target_accuracy=0.87,
    verbose=True, new_run=True
)

state_size = 3
action_size = len(action_space)

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=0.001,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9,
    batch_size=32,
    memory_size=1000,
    target_update_freq=5
)

# -----------------------------
# 4️⃣ DQN Training Loop with Metrics
# -----------------------------
episodes = 5
rewards_log = []
best_val_acc = 0.0
best_model_path = "SavedModels/best_model.keras"
final_model_path = "SavedModels/final_model.keras"

# Prepare CSV for storing metrics
os.makedirs("results", exist_ok=True)
metrics_csv = "results/dqn_test_metrics.csv"
with open(metrics_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Accuracy", "Precision", "Recall", "F1"])

for ep in range(episodes):
    print(f"\n🚀 [EPISODE {ep + 1}/{episodes}] ----------------------------")
    
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx = agent.act(state)
        action = action_space[action_idx]

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action_idx, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        # Save best model
        if info["val_accuracy"] > best_val_acc:
            best_val_acc = info["val_accuracy"]
            env.model.save(best_model_path)
            print(f"💾 [SAVE] New best model (val_acc={best_val_acc:.4f})")

    # Evaluate metrics on validation set after each episode
    y_val_pred_probs = env.model.predict(X_val, verbose=0)
    y_val_pred = (y_val_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    # Save metrics to CSV
    with open(metrics_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ep+1, acc, prec, rec, f1])

    print(f"[EPISODE {ep + 1}] Total Reward = {total_reward:.2f} | Best Val Acc = {best_val_acc:.4f}")
    print(f"📊 Validation Metrics — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    rewards_log.append(total_reward)

    # Save final model at the end of episode
    env.model.save(final_model_path)

print("\n✅ DQN Training completed successfully!")
print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
print(f"💾 Final model saved to {final_model_path}")

# -----------------------------
# 5️⃣ Plot rewards and metrics
# -----------------------------
plot_training_logs(folder_path="results", save_dir="results/plots")



# Updated with minial change
import os
import pickle
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Preprocessing.feature_extraction import load_and_preprocess_imdb
from environment.sentiment_env import SentimentEnv
from models.rl_agent import DQNAgent
from utils.plot_utils import plot_training_logs
import pickle
import os

# Paths to save best results from Phase 1
best_params_path = "SavedModels/best_hparams.pkl"
best_model_path = "SavedModels/best_model.keras"

# Create folder if not exist
os.makedirs("SavedModels", exist_ok=True)
# -----------------------------
# 1️⃣ Load & Preprocess Dataset
# -----------------------------
print("[INFO] Loading and preprocessing IMDB dataset...")
X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# Split for validation
X_val, y_val = X_train[:5000], y_train[:5000]
X_train, y_train = X_train[5000:], y_train[5000:]

os.makedirs("models", exist_ok=True)
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved to models/tokenizer.pkl")

# -----------------------------
# 2️⃣ Define Action Space
# -----------------------------
lr_values = [0.001, 0.0008, 0.0005]
batch_sizes = [64, 128, 256]
dropouts = [0.5]

action_space = [
    {"lr": lr, "batch_size": bs, "dropout": dr}
    for lr in lr_values
    for bs in batch_sizes
    for dr in dropouts
]
print(f"[INFO] Action space size: {len(action_space)}")

# -----------------------------
# 3️⃣ Initialize Environment + Agent
# -----------------------------
env = SentimentEnv(
    X_train, y_train, X_val, y_val,
    step_epochs=3,
    max_steps=10,
    target_accuracy=0.87,
    verbose=True, new_run=True
)

state_size = 3
action_size = len(action_space)

agent = DQNAgent(
    state_size=state_size,
    action_size=action_size,
    lr=0.001,
    gamma=0.95,
    epsilon=1.0,
    epsilon_min=0.05,
    epsilon_decay=0.9,
    batch_size=32,
    memory_size=1000,
    target_update_freq=5
)

# -----------------------------
# 4️⃣ DQN Training Loop with Metrics
# -----------------------------
episodes = 5
rewards_log = []
best_val_acc = 0.0
best_model_path = "SavedModels/best_model.keras"
final_model_path = "SavedModels/final_model.keras"

# Prepare CSV for storing metrics
os.makedirs("results", exist_ok=True)
metrics_csv = "results/dqn_test_metrics.csv"
with open(metrics_csv, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Episode", "Accuracy", "Precision", "Recall", "F1"])

for ep in range(episodes):
    print(f"\n🚀 [EPISODE {ep + 1}/{episodes}] ----------------------------")
    
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action_idx = agent.act(state)
        action = action_space[action_idx]

        next_state, reward, done, info = env.step(action)
        agent.remember(state, action_idx, reward, next_state, done)
        agent.replay()

        state = next_state
        total_reward += reward

        # Save best model
        if info["val_accuracy"] > best_val_acc:
            best_val_acc = info["val_accuracy"]
            best_hparams = info["hyperparams"]

            # ✅ Save best model weights
            env.model.save(best_model_path)

            # ✅ Save best hyperparameters for Phase 2
            with open(best_params_path, "wb") as f:
                pickle.dump(best_hparams, f)

            print(f"💾 [SAVE] New best model (val_acc={best_val_acc:.4f}) with params {best_hparams}")


    # Evaluate metrics on validation set after each episode
    y_val_pred_probs = env.model.predict(X_val, verbose=0)
    y_val_pred = (y_val_pred_probs > 0.5).astype(int)

    acc = accuracy_score(y_val, y_val_pred)
    prec = precision_score(y_val, y_val_pred)
    rec = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)

    # Save metrics to CSV
    with open(metrics_csv, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([ep+1, acc, prec, rec, f1])

    print(f"[EPISODE {ep + 1}] Total Reward = {total_reward:.2f} | Best Val Acc = {best_val_acc:.4f}")
    print(f"📊 Validation Metrics — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
    rewards_log.append(total_reward)

    # Save final model at the end of episode
    env.model.save(final_model_path)

print("\n✅ DQN Training completed successfully!")
print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
print(f"💾 Final model saved to {final_model_path}")

# -----------------------------
# 5️⃣ Plot rewards and metrics
# -----------------------------
plot_training_logs(folder_path="results", save_dir="results/plots")

# Updated with after charge gain 
# import os
# import pickle
# import csv
# import numpy as np
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# from Preprocessing.feature_extraction import load_and_preprocess_imdb
# from environment.sentiment_env import SentimentEnv
# from models.rl_agent import DQNAgent
# from utils.plot_utils import plot_training_logs

# # -----------------------------
# # 1️⃣ Load & Preprocess Dataset
# # -----------------------------
# print("[INFO] Loading and preprocessing IMDB dataset...")
# X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# # Split for validation
# X_val, y_val = X_train[:5000], y_train[:5000]
# X_train, y_train = X_train[5000:], y_train[5000:]

# # Ensure folders exist
# os.makedirs("models", exist_ok=True)
# os.makedirs("SavedModels", exist_ok=True)
# os.makedirs("results", exist_ok=True)

# with open("SavedModels/tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)
# print("✅ Tokenizer saved to SavedModels/tokenizer.pkl")

# # -----------------------------
# # 2️⃣ Define Action Space
# # -----------------------------
# lr_values = [0.001, 0.0008, 0.0005, 0.0003]
# batch_sizes = [64, 128, 256]
# dropouts = [0.2, 0.3, 0.5]
# l2_regs = [0.0, 0.001, 0.005]
# dense_units_values = [32, 64]
# lstm_units_values = [32, 64]

# action_space = [
#     {
#         "lr": lr,
#         "batch_size": bs,
#         "dropout": dr,
#         "l2_reg": l2,
#         "dense_units": du,
#         "lstm_units": lu
#     }
#     for lr in lr_values
#     for bs in batch_sizes
#     for dr in dropouts
#     for l2 in l2_regs
#     for du in dense_units_values
#     for lu in lstm_units_values
# ]
# print(f"[INFO] Action space size: {len(action_space)}")

# # -----------------------------
# # 3️⃣ Initialize Environment + Agent
# # -----------------------------
# env = SentimentEnv(
#     X_train, y_train, X_val, y_val,
#     step_epochs=1,
#     max_steps=10,
#     target_accuracy=0.97,
#     verbose=True, new_run=True
# )

# state_size = 3
# action_size = len(action_space)

# agent = DQNAgent(
#     state_size=state_size,
#     action_size=action_size,
#     lr=0.001,
#     gamma=0.95,
#     epsilon=1.0,
#     epsilon_min=0.05,
#     epsilon_decay=0.9,
#     batch_size=32,
#     memory_size=1000,
#     target_update_freq=5
# )

# # -----------------------------
# # 4️⃣ DQN Training Loop
# # -----------------------------
# episodes = 5
# rewards_log = []
# best_val_acc = 0.0
# best_model_path = "SavedModels/best_model.keras"
# final_model_path = "SavedModels/final_model.keras"

# metrics_csv = "results/dqn_test_metrics.csv"
# with open(metrics_csv, "w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Episode", "Accuracy", "Precision", "Recall", "F1"])

# for ep in range(episodes):
#     print(f"\n🚀 [EPISODE {ep + 1}/{episodes}] ----------------------------")
#     state = env.reset()
#     done = False
#     total_reward = 0

#     while not done:
#         action_idx = agent.act(state)
#         action = action_space[action_idx]

#         next_state, reward, done, info = env.step(action)
#         agent.remember(state, action_idx, reward, next_state, done)
#         agent.replay()

#         state = next_state
#         total_reward += reward

#         # Save best model
#         if info["val_accuracy"] > best_val_acc:
#             best_val_acc = info["val_accuracy"]
#             env.model.save(best_model_path)
#             print(f"💾 [SAVE] New best model (val_acc={best_val_acc:.4f})")

#     # Evaluate metrics
#     y_val_pred_probs = env.model.predict(X_val, verbose=0)
#     y_val_pred = (y_val_pred_probs > 0.5).astype(int)

#     acc = accuracy_score(y_val, y_val_pred)
#     prec = precision_score(y_val, y_val_pred)
#     rec = recall_score(y_val, y_val_pred)
#     f1 = f1_score(y_val, y_val_pred)

#     with open(metrics_csv, "a", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow([ep+1, acc, prec, rec, f1])

#     print(f"[EPISODE {ep + 1}] Total Reward = {total_reward:.2f} | Best Val Acc = {best_val_acc:.4f}")
#     print(f"📊 Validation Metrics — Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
#     rewards_log.append(total_reward)

#     # Save final model after episode
#     env.model.save(final_model_path)

# print("\n✅ DQN Training completed successfully!")
# print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
# print(f"💾 Final model saved to {final_model_path}")

# # -----------------------------
# # 5️⃣ Plot rewards and metrics
# # -----------------------------
# plot_training_logs(folder_path="results", save_dir="results/plots")
