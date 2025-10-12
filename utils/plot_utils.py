import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_training_logs(folder_path="results", save_dir="results/plots", prefix="accuracy_logs"):
    """
    Loads the latest CSV file in folder_path that starts with the given prefix
    and plots training vs validation loss and accuracy in a single figure.
    """
    # Find CSV files that start with the prefix
    csv_files = [os.path.join(folder_path, f)
                 for f in os.listdir(folder_path)
                 if f.endswith(".csv") and f.startswith(prefix)]

    if not csv_files:
        print(f"[WARN] No CSV files starting with '{prefix}' found in {folder_path}.")
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

    # Create a single figure with 2 subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 5))

    # Plot Loss
    axs[0].plot(epochs, train_loss, label="Training Loss", marker="o")
    axs[0].plot(epochs, val_loss, label="Validation Loss", marker="s")
    axs[0].set_title("Training vs Validation Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Loss")
    axs[0].legend()
    axs[0].grid(True)

    # Plot Accuracy
    axs[1].plot(epochs, train_acc, label="Training Accuracy", marker="o")
    axs[1].plot(epochs, val_acc, label="Validation Accuracy", marker="s")
    axs[1].set_title("Training vs Validation Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    axs[1].grid(True)

    # Adjust layout
    plt.tight_layout()

    # Save the combined figure
    combined_path = os.path.join(save_dir, "loss_accuracy_plot.png")
    plt.savefig(combined_path)
    plt.show()
    print(f"📊 Combined Loss & Accuracy plot saved to {combined_path}")
