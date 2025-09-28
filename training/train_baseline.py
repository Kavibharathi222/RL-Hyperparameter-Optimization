from Preprocessing.feature_extraction import load_and_preprocess_imdb 
from models.baseline_model import build_baseline_model
from sklearn.model_selection import train_test_split

def train_baseline():
    # Load dataset
    X_train, y_train, X_test, y_test = load_and_preprocess_imdb()

    # Split validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Build model
    model = build_baseline_model()

    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=5,
        batch_size=64,
        validation_data=(X_val, y_val),
        verbose=1
    )

    # Evaluate model
    loss, acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {acc:.4f}")

    return history, model

if __name__ == "__main__":
    train_baseline()
