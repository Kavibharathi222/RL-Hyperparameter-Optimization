import os
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load CSV
# -----------------------------
csv_path = "results/accuracy_logs.csv"   # <-- change if needed
df = pd.read_csv(csv_path)

print("Columns in CSV:", df.columns.tolist())

# -----------------------------
# 2. Extract columns correctly
# -----------------------------
epochs = df["epoch"]
train_loss = df["train_loss"]
val_loss = df["val_loss"]
train_acc = df["train_accuracy"]
val_acc = df["val_accuracy"]

# -----------------------------
# 3. Create output folder
# -----------------------------
os.makedirs("results/plots", exist_ok=True)

# -----------------------------
# 4. Plot Training vs Validation Loss
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_loss, label="Training Loss", marker="o")
plt.plot(epochs, val_loss, label="Validation Loss", marker="s")
plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.savefig("results/plots/loss_plot.png")
plt.show()

# -----------------------------
# 5. Plot Training vs Validation Accuracy
# -----------------------------
plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, label="Training Accuracy", marker="o")
plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.savefig("results/plots/accuracy_plot.png")
plt.show()

plt.close()
