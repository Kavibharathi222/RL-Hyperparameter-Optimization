import matplotlib.pyplot as plt
import pandas as pd
import os 

def plot_training_history(history=None, log_file="results/baseline_logs.csv"):
    """
    Plots training and validation accuracy/loss curves.
    Reads from CSV if history not provided.
    """
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
    os.makedirs("results/plots", exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()
    
    plt.savefig("results/plots/Base.png")  
    plt.close()
