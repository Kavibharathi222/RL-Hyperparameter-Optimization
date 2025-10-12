






import os
import pickle
import matplotlib.pyplot as plt
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
# 2️⃣ Define Action Space (Scaling-based)
# -----------------------------
# -----------------------------
# 2️⃣ Define Discrete Action Space (Stable Version)
# -----------------------------
lr_values = [0.001, 0.0008, 0.0005]
batch_sizes = [64, 128, 256]
dropouts = 0.5

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
# 4️⃣ DQN Training Loop
# -----------------------------
episodes = 5
rewards_log = []
best_val_acc = 0.0
numofepoch =0
best_model_path = "SavedModels/best_model.keras"
final_model_path = "SavedModels/final_model.keras"

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

        if info["val_accuracy"] > best_val_acc:
            best_val_acc = info["val_accuracy"]
            env.model.save(best_model_path)
            print(f"💾 [SAVE] New best model (val_acc={best_val_acc:.4f})")

        if done:
            print("🛑 Training stopped (done=True). Saving current model...")
            env.model.save(final_model_path)
            break

    print(f"[EPISODE {ep + 1}] Total Reward = {total_reward:.2f} | Best Val Acc = {best_val_acc:.4f}")
    rewards_log.append(total_reward)

print("\n✅ Training completed successfully!")
print(f"🏆 Best validation accuracy: {best_val_acc:.4f}")
env.model.save(final_model_path)
print(f"💾 Final model saved to {final_model_path}")

plot_training_logs(folder_path="results", save_dir="results/plots")
# -----------------------------
# 5️⃣ Plot Rewards per Episode
# -----------------------------
# import os
# import pandas as pd
# import matplotlib.pyplot as plt

# # Read your log file (CSV or TSV). Adjust delimiter if needed ("," or "\t")
# log_file = "results/accuracy_logs.csv"   # update with your actual file path
# df = pd.read_csv(log_file)
# print(df.columns.tolist())
# print(df.head())

# # Extract columns
# epochs = df["epoch"]
# train_loss = df["train_loss"]
# val_loss = df["val_loss"]
# train_acc = df["train_accuracy"]
# val_acc = df["val_accuracy"]

# # Create folder if not exists
# os.makedirs("results/plots", exist_ok=True)

# # --- Plot 1: Training vs Validation Loss ---
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_loss, label="Training Loss", marker="o")
# plt.plot(epochs, val_loss, label="Validation Loss", marker="s")
# plt.title("Training vs Validation Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.grid(True)
# plt.legend()
# plt.savefig("results/plots/loss_plot.png")
# plt.show()

# # --- Plot 2: Training vs Validation Accuracy ---
# plt.figure(figsize=(8, 5))
# plt.plot(epochs, train_acc, label="Training Accuracy", marker="o")
# plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
# plt.title("Training vs Validation Accuracy")
# plt.xlabel("Epoch")
# plt.ylabel("Accuracy")
# plt.grid(True)
# plt.legend()
# plt.savefig("results/plots/accuracy_plot.png")
# plt.show()

# plt.close()