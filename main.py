from training.train_baseline import train_baseline
# from utils.visualization import plot_training_history
from utils.graph import plot_training_history

if __name__ == "__main__":
    history, model = train_baseline()
    plot_training_history(history)
