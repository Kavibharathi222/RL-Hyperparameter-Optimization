import pandas as pd
import matplotlib.pyplot as plt
import os 

def plot_training_logs():
    # Hardcoded CSV file path
    csv_file = r"D:\MiniProject\Project\RL-Hyperparameter-Optimization\results\accuracy_logs.csv"
    os.makedirs("results/plots", exist_ok=True)
    # Read the CSV
    logs = pd.read_csv(csv_file)

    print("Available columns:", logs.columns.tolist())  # Debug

    epochs = logs["epoch"]

    # Create subplots: one for accuracy, one for loss
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy plot
    axes[0].plot(epochs, logs["train_accuracy"], label="Train Accuracy", marker='o')
    axes[0].plot(epochs, logs["val_accuracy"], label="Validation Accuracy", marker='o')
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Accuracy")
    axes[0].set_title("Training Accuracy vs Validation Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss plot
    axes[1].plot(epochs, logs["train_loss"], label="Train Loss", marker='o')
    axes[1].plot(epochs, logs["val_loss"], label="Validation Loss", marker='o')
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Loss vs Validation Loss")
    axes[1].legend()
    axes[1].grid(True)



    plt.tight_layout()
    
    plt.savefig("results/plots/example_plot.png") 
    plt.show()
    print("It is save in Visualization")
    plt.close()
# Usage
plot_training_logs()

# import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training History')
    plt.show()
