import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from Preprocessing.feature_extraction import load_and_preprocess_imdb
from environment.sentiment_env import SentimentEnv

# ===============================
# 1️⃣ Load and Preprocess Dataset
# ===============================
print("🔹 Loading and preprocessing IMDB dataset...")
X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# Split training into train/validation
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ===================================
# 2️⃣ Initialize RL Environment
# ===================================
print("🔹 Initializing RL Environment...")
env = SentimentEnv(
    X_tr,
    y_tr,
    X_val,
    y_val,
    step_epochs=1,   # how many epochs per step
    max_steps=5,     # total RL steps (you can increase)
    verbose=True
)

# Reset environment
state = env.reset()

# ===================================
# 3️⃣ Manual RL Action Loop
# ===================================
print("🔹 Starting RL Agent training loop...")

actions = [
    {"lr": 1e-3, "batch_size": 64},
    {"lr": 5e-4},                  # lower learning rate
    {"dropout": 0.6},              # change dropout
    {"lr": 1e-4, "batch_size": 32},
    {"stop": True},                # stop manually
]

for a in actions:
    next_state, reward, done, info = env.step(a)
    print("Step result:", info)
    if done:
        break

print("✅ RL Training completed successfully!")

# ===================================
# 4️⃣ Save Model and Tokenizer
# ===================================
os.makedirs("results", exist_ok=True)

model_path = "results/rl_trained_model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(env.model, f)
print(f"✅ Model saved at: {model_path}")

tokenizer_path = "results/tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"✅ Tokenizer saved at: {tokenizer_path}")

# ===================================
# 5️⃣ Plot Metrics from CSV
# ===================================
csv_path = "results/accuracy_logs.csv"
if os.path.exists(csv_path):
    logs = pd.read_csv(csv_path)

    plt.figure(figsize=(8, 5))
    plt.plot(logs['epoch'], logs['val_accuracy'], label="Validation Accuracy")
    plt.plot(logs['epoch'], logs['reward'], label="Reward")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("RL Hyperparameter Optimization Progress")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("⚠️ Log file not found. Please ensure training completed successfully.")
