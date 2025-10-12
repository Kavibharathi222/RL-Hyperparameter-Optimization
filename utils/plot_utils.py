import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_logs(folder_path="results", save_dir="results/plots"):
    """
    Automatically loads the latest CSV file in the given folder
    and plots training vs validation loss and accuracy.
    """
    # Find all CSV files in the folder
    csv_files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.endswith(".csv")]

    if not csv_files:
        print(f"[WARN] No CSV files found in {folder_path}.")
        return

    # Sort by modification time and pick the latest
    csv_files.sort(key=os.path.getmtime)
    latest_csv = csv_files[-1]
    print(f"📂 Loading latest CSV file: {latest_csv}")

    # Read the CSV
    df = pd.read_csv(latest_csv)

    # Check that required columns exist
    required_cols = {"epoch", "train_loss", "val_loss", "train_accuracy", "val_accuracy"}
    if not required_cols.issubset(df.columns):
        print(f"[ERROR] Missing required columns in {latest_csv}. Found columns: {list(df.columns)}")
        return

    epochs = df["epoch"]
    train_loss = df["train_loss"]
    val_loss = df["val_loss"]
    train_acc = df["train_accuracy"]
    val_acc = df["val_accuracy"]

    os.makedirs(save_dir, exist_ok=True)

    # Plot Loss
    plt.figure(figsize=(8, 5))
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

    # Plot Accuracy
    plt.figure(figsize=(8, 5))
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
