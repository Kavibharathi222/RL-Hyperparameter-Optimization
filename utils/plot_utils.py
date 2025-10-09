# utils/plot_utils.py
import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_logs(csv_path="results/accuracy_logs.csv", save_dir="results/plots"):
    """
    Reads training logs CSV and plots loss & accuracy.
    """
    if not os.path.exists(csv_path):
        print(f"[WARN] CSV file {csv_path} not found. Skipping plotting.")
        return

    df = pd.read_csv(csv_path)

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    train_acc = df["train_accuracy"]
    val_acc = df["val_accuracy"]

    os.makedirs(save_dir, exist_ok=True)

    # Plot loss
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_loss, label="Training Loss", marker="o")
    plt.plot(epochs, val_loss, label="Validation Loss", marker="s")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    loss_path = os.path.join(save_dir, "loss_plot.png")
    plt.savefig(loss_path)
    plt.show()
    print(f"📊 Loss plot saved to {loss_path}")

    # Plot accuracy
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_acc, label="Training Accuracy", marker="o")
    plt.plot(epochs, val_acc, label="Validation Accuracy", marker="s")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    acc_path = os.path.join(save_dir, "accuracy_plot.png")
    plt.savefig(acc_path)
    plt.show()
    print(f"📊 Accuracy plot saved to {acc_path}")
