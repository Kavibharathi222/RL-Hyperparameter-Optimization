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
    max_steps=10,   # RL steps (you can adjust)
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
