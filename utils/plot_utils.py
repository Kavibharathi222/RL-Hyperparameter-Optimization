import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_logs(folder_path="results",
                       save_dir="results/plots",
                       prefix="accuracy_logs",
                       model=None,
                       X_test=None,
                       y_test=None):
    """
    Loads the latest CSV file in folder_path that starts with the given prefix
    and plots training vs validation loss and accuracy in a single figure.
    If model and test data are provided, also plots the confusion matrix.
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

    # Adjust layout and save plot
    plt.tight_layout()
    combined_path = os.path.join(save_dir, "loss_accuracy_plot.png")
    plt.savefig(combined_path, dpi=300)
    plt.show()
    print(f"📊 Combined Loss & Accuracy plot saved to {combined_path}")

    # -----------------------------
    # ✅ Confusion Matrix Section
    # -----------------------------
    if model is not None and X_test is not None and y_test is not None:
        print("[INFO] Generating confusion matrix...")

        y_pred = model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype("int32")

        cm = confusion_matrix(y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                      display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - FineTune")

        cm_path = os.path.join(save_dir, "confusion_finetune.png")
        plt.savefig(cm_path, dpi=300)
        plt.show()

        print(f"✅ Confusion matrix saved to {cm_path}")
    else:
        print("[WARN] Confusion matrix not plotted (model or test data missing).")
