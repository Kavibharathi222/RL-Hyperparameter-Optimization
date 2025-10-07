# from Preprocessing.feature_extraction import load_and_preprocess_imdb
# from sklearn.model_selection import train_test_split
# from environment.sentiment_env import SentimentEnv
# import matplotlib.pyplot as plt
# import pandas as pd
# import pickle

# print("🔹 Loading and preprocessing IMDB dataset...")
# print("🔹 Loading and preprocessing IMDB dataset...")
# X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

# X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# print("🔹 Initializing RL Environment...")
# env = SentimentEnv(X_tr, y_tr, X_val, y_val, step_epochs=1, max_steps=30,target_accuracy=86, verbose=True)

# state = env.reset()     

# print("🔹 Starting RL Agent training loop...")
# for step in range(env.max_steps):
#     action = {}

#     # Example simple adjustments (replace with RL agent’s policy)
#     if step % 5 == 0:
#         action["lr"] = float(max(env.current_hyperparams["lr"] * 0.5, 1e-6))
#     if step % 10 == 0:
#         action["dropout"] = float(min(env.current_hyperparams["dropout"] + 0.1, 0.7))

#     next_state, reward, done, info = env.step(action)
#     print(f"Step {step+1}: val_acc={info['val_accuracy']:.4f}, reward={reward:.4f}, hyperparams={info['hyperparams']}")

#     if done:
#         break

# # # Save final model and tokenizer
# # print("🔹 Saving model...")
# # env.model.save("results/final_rl_model.h5")
# # Save final model, hyperparams, and tokenizer
# print("🔹 Saving model and tokenizer...")
# env.model.save("results/final_rl_model.h5")

# with open("results/final_hyperparams.pkl", "wb") as f:
#     pickle.dump(env.current_hyperparams, f)

# with open("results/final_tokenizer.pkl", "wb") as f:
#     pickle.dump(tokenizer, f)


# with open("results/final_hyperparams.pkl", "wb") as f:
#     pickle.dump(env.current_hyperparams, f)

# # Plot logs
# try:
#     logs = pd.read_csv("results/accuracy_logs.csv")
#     plt.plot(logs["epoch"], logs["val_accuracy"], label="Validation Accuracy")
#     plt.plot(logs["epoch"], logs["reward"], label="Reward")
#     plt.xlabel("Epoch")
#     plt.ylabel("Value")
#     plt.legend()
#     plt.title("RL Training Progress")
#     plt.show()
# except Exception as e:
#     print("Could not plot logs:", e)



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

# Ensure your preprocessing returns tokenizer
X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)

X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# ===============================
# 2️⃣ Initialize Environment
# ===============================
print("🔹 Initializing RL Environment...")
env = SentimentEnv(
    X_tr, y_tr,
    X_val, y_val,
    step_epochs=1,          # train 3 epochs per step
    max_steps=10,          # up to 300 steps
    target_accuracy=0.87,   # stop if 90% validation accuracy
    verbose=True
)

state = env.reset()

previous_acc = 0.0
plateau_counter = 0

# ===============================
# 3️⃣ Adaptive RL Loop
# ===============================
print("🔹 Starting RL Agent training loop...")

for step in range(env.max_steps):
    prev_val_acc = float(state[0])
    prev_val_loss = float(state[1])
    minstep=6
    action = {}

    # === CONDITIONS ===

    # Condition 1: Validation loss ↑ → reduce LR
    if prev_val_loss > env.prev_val_loss:
        action["lr"] = float(max(float(env.current_hyperparams["lr"]) * 0.5, 1e-6))

    # Condition 3: Overfitting → increase dropout
    if (env.prev_val_acc < env.best_val_acc) and (env.prev_val_loss < 0.5):
        action["dropout"] = float(min(float(env.current_hyperparams["dropout"]) + 0.1, 0.7))

    # Condition 4: Underfitting → decrease dropout or increase LR
    if prev_val_acc < 0.65 and prev_val_loss > 0.7:
        action["dropout"] = float(max(float(env.current_hyperparams["dropout"]) - 0.1, 0.1))
        action["lr"] = float(min(float(env.current_hyperparams["lr"]) * 1.2, 1e-2))

    # Condition 5: Plateau in accuracy → adjust
    if abs(prev_val_acc - previous_acc) < 0.001:
        plateau_counter += 1
    else:
        plateau_counter = 0
    if plateau_counter >= 3:
        action["lr"] = float(env.current_hyperparams["lr"]) * 0.7
        action["batch_size"] = int(min(env.current_hyperparams["batch_size"] * 2, 256))
        plateau_counter = 0

    # Condition 6: Scheduled adjustment every 50 epochs
    if env.epoch > 0 and env.epoch % 50 == 0:
        action["lr"] = float(max(float(env.current_hyperparams["lr"]) * 0.8, 1e-6))

    # Default: keep batch size same
    action.setdefault("batch_size", int(env.current_hyperparams["batch_size"]))

    # === Step ===
    state, reward, done, info = env.step(action)
    previous_acc = float(info["val_accuracy"])

    print(f"Step {step+1} | Epoch {info['epoch']} | "
          f"ValAcc: {info['val_accuracy']:.4f} | ValLoss: {info['val_loss']:.4f} | "
          f"Reward: {reward:.4f} | Params: {info['hyperparams']}")

    if done and step > minstep :
        print("✅ Training stopped (target accuracy or max steps reached).")
        break

print("✅ RL Training completed successfully!")

# ===============================
# 4️⃣ Save Model + Tokenizer
# ===============================
os.makedirs("results", exist_ok=True)

# Save model as Pickle
model_pickle_path = "results/rl_trained_model.pkl"
with open(model_pickle_path, "wb") as f:
    pickle.dump(env.model, f)
print(f"✅ Model saved (pickle) at: {model_pickle_path}")

# Save model as Keras H5
model_h5_path = "results/rl_trained_model.h5"
env.model.save(model_h5_path)
print(f"✅ Model saved (H5) at: {model_h5_path}")

# Save tokenizer
tokenizer_path = "results/tokenizer.pkl"
with open(tokenizer_path, "wb") as f:
    pickle.dump(tokenizer, f)
print(f"✅ Tokenizer saved at: {tokenizer_path}")

# ===============================
# 5️⃣ Plot Training Logs
# ===============================
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
