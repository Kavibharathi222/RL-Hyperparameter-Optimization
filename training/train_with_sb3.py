from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from Preprocessing.feature_extraction import load_and_preprocess_imdb
from environment.sentiment_env_sb3 import SentimentEnvSB3
import pickle
import os

# -----------------------------
# Load data
# -----------------------------
X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(
    num_words=10000, maxlen=200
)

X_val, y_val = X_train[:5000], y_train[:5000]
X_train, y_train = X_train[5000:], y_train[5000:]

# -----------------------------
# Action space
# -----------------------------
lr_values = [0.001, 0.0008, 0.0005]
batch_sizes = [64, 128]
dropouts = [0.1, 0.3, 0.5]

action_space = [
    {"lr": lr, "batch_size": bs, "dropout": dr}
    for lr in lr_values
    for bs in batch_sizes
    for dr in dropouts
]

# -----------------------------
# Environment
# -----------------------------
env = DummyVecEnv([
    lambda: SentimentEnvSB3(
        X_train, y_train, X_val, y_val,
        action_space_list=action_space,
        max_steps=10
    )
])

# -----------------------------
# DQN Agent (SB3)
# -----------------------------
model = DQN(
    "MlpPolicy",
    env,
    learning_rate=1e-3,
    buffer_size=5000,
    learning_starts=100,
    batch_size=32,
    gamma=0.95,
    target_update_interval=500,
    exploration_fraction=0.3,
    exploration_final_eps=0.05,
    verbose=1
)

# -----------------------------
# Train
# -----------------------------
model.learn(total_timesteps=30)

# -----------------------------
# Save best model
# -----------------------------
os.makedirs("SavedModels", exist_ok=True)
model.save("SavedModels/sb3_dqn_sentiment")

print("✅ SB3 DQN training completed")
