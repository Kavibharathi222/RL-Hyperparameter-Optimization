from Preprocessing.feature_extraction import load_and_preprocess_imdb
from models.baseline_model import build_baseline_model
from sklearn.model_selection import train_test_split
import csv
import os
import pickle

def train_baseline():
    print("🔹 Loading and preprocessing IMDB dataset...")
    X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb()

    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    print("🔹 Building baseline model (BiLSTM)...")
    model = build_baseline_model()

    # Train model
    print("🔹 Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"✅ Test Accuracy: {acc:.4f}")

    # 🔸 Save metrics to CSV
    os.makedirs("results", exist_ok=True)
    log_file = "results/baseline_logs.csv"

    with open(log_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        for epoch in range(len(history.history["accuracy"])):
            writer.writerow([
                epoch + 1,
                history.history["loss"][epoch],
                history.history["accuracy"][epoch],
                history.history["val_loss"][epoch],
                history.history["val_accuracy"][epoch]
            ])
    print(f"📊 Training metrics saved to {log_file}")

    # 🔸 Save model in pickle format
    os.makedirs("models/saved_models", exist_ok=True)
    model_path = "results/baseline_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"✅ Model saved successfully at: {model_path}")

    # Also save tokenizer (important for manual testing)
    tokenizer_path = "results/tokenizer_basemodel.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved successfully at: {tokenizer_path}")

    return history, model

if __name__ == "__main__":
    train_baseline()
