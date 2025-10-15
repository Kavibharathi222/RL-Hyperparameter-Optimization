import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_training_history(history=None, 
                          log_file="results/baseline_logs.csv",
                          model=None,
                          X_test=None,
                          y_test=None):
    """
    Plots training and validation accuracy/loss curves.
    Reads from CSV if history not provided.
    Also plots and saves confusion matrix if model & test data are given.
    """
    # -----------------------------
    # ✅ Load Data (from history or CSV)
    # -----------------------------
    if history:
        acc = history.history["accuracy"]
        val_acc = history.history["val_accuracy"]
        loss = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs = range(1, len(acc) + 1)
    else:
        logs = pd.read_csv(log_file)
        epochs = logs["epoch"]
        acc = logs["train_accuracy"]
        val_acc = logs["val_accuracy"]
        loss = logs["train_loss"]
        val_loss = logs["val_loss"]

    # -----------------------------
    # ✅ Create folder to save plots
    # -----------------------------
    os.makedirs("results/plots", exist_ok=True)

    # -----------------------------
    # ✅ Plot Accuracy & Loss
    # -----------------------------
    plt.figure(figsize=(10, 5))

    # Accuracy Plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.ylim(0.1, 0.99)
    plt.yticks([i/10 for i in range(1, 11)])

    # Loss Plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.ylim(0, 3)

    plt.tight_layout()
    plt.savefig("results/plots/Base.png", dpi=300)
    plt.show()
    print("✅ Training & validation plots saved to results/plots/Base.png")

    # -----------------------------
    # ✅ Plot Confusion Matrix
    # -----------------------------
    if model is not None and X_test is not None and y_test is not None:
        print("[INFO] Generating confusion matrix...")
        y_pred = model.predict(X_test)
        y_pred_classes = (y_pred > 0.5).astype("int32")

        cm = confusion_matrix(y_test, y_pred_classes)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
        disp.plot(cmap="Blues", values_format="d")
        plt.title("Confusion Matrix - Baseline")
        plt.savefig("results/plots/confusion_baseline.png", dpi=300)
        plt.show()
        print("✅ Confusion matrix saved to results/plots/confusion_baseline.png")
    else:
        print("[WARN] Confusion matrix not plotted (model or test data missing).")
