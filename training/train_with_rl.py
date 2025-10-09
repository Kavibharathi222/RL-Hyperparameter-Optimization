import os
import pickle
import matplotlib.pyplot as plt
from Preprocessing.feature_extraction import load_and_preprocess_imdb
from environment.sentiment_env import SentimentEnv
from models.rl_agent import DQNAgent


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
lr_factors = [0.5, 0.8, 1.0, 1.2, 1.5]
batch_factors = [0.5, 1.0, 1.5, 2.0]
dropout_factors = [0.8, 1.0, 1.2]

action_space = [
    {"lr": lr, "batch_size": bs, "dropout": dr}
    for lr in lr_factors
    for bs in batch_factors
    for dr in dropout_factors
]
print(f"[INFO] Action space size: {len(action_space)}")


# -----------------------------
# 3️⃣ Initialize Environment + Agent
# -----------------------------
env = SentimentEnv(
    X_train, y_train, X_val, y_val,
    step_epochs=1,
    max_steps=2,
    target_accuracy=0.87,
    verbose=True
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
episodes = 1
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


# -----------------------------
# 5️⃣ Plot Rewards per Episode
# -----------------------------
os.makedirs("results/plots", exist_ok=True)
# step	epoch	train_loss	train_accuracy"	val_loss	val_accuracy	reward	best_val_accuracy	learning_rate	droupout	droupout

# --- Plot 1: Training vs Validation Loss ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training Loss", marker="o")
plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation Loss", marker="s")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("results/plots/loss_plot.png")
plt.show()

# --- Plot 2: Training vs Validation Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(range(1, len(train_accuracy) + 1), train_accuracy, label="Training Accuracy", marker="o")
plt.plot(range(1, len(val_accuracy) + 1), val_accuracy, label="Validation Accuracy", marker="s")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("results/plots/accuracy_plot.png")
plt.show()


plt.close()