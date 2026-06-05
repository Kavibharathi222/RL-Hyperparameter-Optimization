import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Hide TF/Keras deprecation warnings

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

# -----------------------------
# 1️⃣ Rebuild a lightweight RL model
# -----------------------------
def build_lightweight_rl_model(vocab_size=10000, maxlen=200, embedding_dim=64, lstm_units=32):
    """
    Lighter model for faster inference:
    - Smaller embedding dimension
    - Smaller LSTM units
    - Unidirectional LSTM instead of bidirectional
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
    model.add(LSTM(lstm_units))  # unidirectional for speed
    model.add(Dense(1, activation='sigmoid'))
    return model

# -----------------------------
# 2️⃣ Load tokenizer
# -----------------------------
with open("results/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1
maxlen = 200
rl_model = build_lightweight_rl_model(vocab_size=vocab_size, maxlen=maxlen)

# -----------------------------
# 3️⃣ Load trained weights (ensure saved with same architecture)
# -----------------------------
# ⚠ Make sure weights were saved with compatible architecture
rl_model.load_weights("SavedModels/final_model.keras")

# -----------------------------
# 4️⃣ Warm-up the model (optional, speeds first prediction)
# -----------------------------
dummy_input = np.zeros((1, maxlen))
rl_model.predict(dummy_input, verbose=0)

# -----------------------------
# 5️⃣ Prediction function
# -----------------------------
def predict_sentiment(model, tokenizer, text, maxlen=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if pred >= 0.5 else "Negative 😞"
    return label, float(pred)

# -----------------------------
# 6️⃣ Test sample
# -----------------------------
sample_text = "I really loved this movie, it was fantastic!"
rl_pred, rl_score = predict_sentiment(rl_model, tokenizer, sample_text)
print("RL-Trained Lightweight Model Prediction:", rl_pred, "(score:", rl_score, ")")
