from Preprocessing.feature_extraction import load_and_preprocess_imdb
from sklearn.model_selection import train_test_split
from environment.sentiment_env import SentimentEnv

# load + split
X_train, y_train, X_test, y_test = load_and_preprocess_imdb(num_words=10000, maxlen=200)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

env = SentimentEnv(X_tr, y_tr, X_val, y_val, step_epochs=1, max_steps=2, verbose=True)

state = env.reset()

# manual policy loop example:
actions = [
    {"lr": 1e-3, "batch_size": 64},
    {"lr": 5e-4},                      # decrease lr
    {"dropout": 0.6},                  # increase dropout (rebuild & restore weights)
    {"lr": 1e-4, "batch_size": 32},
    {"stop": True},                    # request stop
]

for a in actions:
    next_state, reward, done, info = env.step(a)
    print("Step result:", info)
    if done:
        break
# after training loop ends
import matplotlib.pyplot as plt
import pandas as pd

logs = pd.read_csv("results/accuracy_logs.csv")

plt.plot(logs['epoch'], logs['val_accuracy'], label="Validation Accuracy")
plt.plot(logs['epoch'], logs['reward'], label="Reward")
plt.xlabel("Epoch")
plt.ylabel("Value")
plt.legend()
plt.title("RL Hyperparameter Optimization Progress")
plt.show()
