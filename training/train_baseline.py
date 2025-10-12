from Preprocessing.feature_extraction import load_and_preprocess_imdb
from models.baseline_model import build_baseline_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import csv
import os
import pickle
import numpy as np

def train_baseline():
    print("🔹 Loading and preprocessing IMDB dataset...")
    X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb()

    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )

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

    # -------------------------------
    # 🔍 Find optimal threshold on validation set
    # -------------------------------
    print("\n🔹 Finding optimal threshold for best F1-score...")
    y_val_probs = model.predict(X_val, verbose=0)
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_t, best_f1 = 0.5, 0.0

    for t in thresholds:
        preds = (y_val_probs > t).astype(int)
        f1 = f1_score(y_val, preds)
        if f1 > best_f1:
            best_t, best_f1 = t, f1

    print(f"✅ Optimal threshold found: {best_t:.2f} (Best F1: {best_f1:.4f})")

    # -------------------------------
    # Evaluate model on test data using optimal threshold
    # -------------------------------
    print("\n🔹 Evaluating on test data...")
    y_test_probs = model.predict(X_test, verbose=0)
    y_pred = (y_test_probs > best_t).astype("int32")

    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)

    print("\n📊 Detailed Test Metrics (using optimal threshold):")
    print(f"Accuracy : {test_accuracy:.4f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall   : {test_recall:.4f}")
    print(f"F1 Score : {test_f1:.4f}")
    print(f"🔸 Threshold used: {best_t:.2f}")

    # -------------------------------
    # Save training metrics to CSV
    # -------------------------------
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
    print(f"✅ Training metrics saved to {log_file}")

    # -------------------------------
    # Save model and tokenizer
    # -------------------------------
    os.makedirs("SavedModels", exist_ok=True)
    model_path = "SavedModels/baseline_model.keras"
    model.save(model_path)
    print(f"✅ Model saved successfully at: {model_path}")

    tokenizer_path = "SavedModels/tokenizer_basemodel.pkl"
    with open(tokenizer_path, "wb") as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved successfully at: {tokenizer_path}")

    # -------------------------------
    # Save test metrics + best threshold
    # -------------------------------
    metrics_path = "results/test_metrics.csv"
    with open(metrics_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Accuracy", "Precision", "Recall", "F1", "Best_Threshold"])
        writer.writerow([test_accuracy, test_precision, test_recall, test_f1, best_t])
    print(f"📁 Test metrics (with threshold) saved to {metrics_path}")

    return history, model

if __name__ == "__main__":
    train_baseline()
