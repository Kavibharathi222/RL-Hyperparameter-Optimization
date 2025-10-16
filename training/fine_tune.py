import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from Preprocessing.feature_extraction import load_and_preprocess_imdb
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
# -----------------------------
# 1️⃣ Load data
# -----------------------------
print("Actual Training is Started ")
def training():
    print("[INFO] Loading and preprocessing IMDB dataset...")
    X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_imdb(num_words=10000, maxlen=200)
    X_val, y_val = X_train[:5000], y_train[:5000]
    X_train, y_train = X_train[5000:], y_train[5000:]

    # -----------------------------
    # 2️⃣ Load best hyperparameters and model
    # -----------------------------
    with open("SavedModels/best_hparams.pkl", "rb") as f:
        best_hparams = pickle.load(f)

    print(f"✅ Loaded best hyperparameters: {best_hparams}")

    model = Sequential([
        Embedding(input_dim=10000, output_dim=128, input_length=200),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(best_hparams["dropout"]),
        Dense(64, activation="relu"),
        Dropout(best_hparams["dropout"]),
        Dense(1, activation="sigmoid")
    ])

    opt = Adam(learning_rate=best_hparams["lr"])
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])

    # Build model before loading weights
    model.build((None, 200))

    # Load the best weights from Phase 1
    model.load_weights("SavedModels/best_model.keras")

    # -----------------------------
    # 3️⃣ Fine-tune (stable training)
    # -----------------------------
    print("[INFO] Starting fine-tuning with fixed hyperparameters...")
    early_stop = EarlyStopping(
        monitor='val_loss',       # Track validation loss
        patience=8,               # Stop after 5 epochs without improvement
        restore_best_weights=True # Roll back to best weights
    )

    # -----------------------------
    # 3️⃣ Fine-tune (stable training with early stopping)
    # -----------------------------
    print("[INFO] Starting fine-tuning with Early Stopping...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=best_hparams["batch_size"],
        verbose=1,
        callbacks=[early_stop]
    )

    # -----------------------------
    # 4️⃣ Save the final fine-tuned model
    # -----------------------------
    model.save("SavedModels/fine_tuned_model.keras")
    print("✅ Fine-tuning complete. Model saved to SavedModels/fine_tuned_model.keras")

    # -----------------------------
    # 5️⃣ Plot Training Curves (Accuracy & Loss)
    # -----------------------------
    plt.figure(figsize=(10, 5))

    # --- Accuracy subplot ---
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training vs Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.ylim(0.1, 0.99)
    plt.yticks([i/10 for i in range(1, 11)])  # 0.1 to 1.0

    # --- Loss subplot ---
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training vs Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 3)

    plt.tight_layout()
    plt.savefig("results/plots/fine_tune_performance.png", dpi=300)
    plt.show()

    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

    print("[INFO] Generating confusion matrix...")

    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred_classes)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negative", "Positive"])
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix - FineTune")
    plt.savefig("results/plots/confusion_finetune.png", dpi=300)
    plt.show()

    print("✅ Confusion matrix saved to results/plots/confusion_finetune.png")
    print("📊 Plot saved to results/plots/fine_tune_performance.png")
    precision = precision_score(y_test, y_pred_classes)
    recall = recall_score(y_test, y_pred_classes)
    f1 = f1_score(y_test, y_pred_classes)
    accuracy = accuracy_score(y_test, y_pred_classes)

    print("\n📈 Model Performance on Test Data:")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print(f"Accuracy  : {accuracy:.4f}")
