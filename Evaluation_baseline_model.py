import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Hide TF/Keras deprecation warnings

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Load baseline model + tokenizer
# -------------------------
baseline_model = tf.keras.models.load_model("SavedModels/baseline_model.h5")
print("Baseline model input shape:", baseline_model.input_shape)

with open("SavedModels/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# Helper function
# -------------------------
def predict_sentiment(model, text, tokenizer, maxlen=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if pred >= 0.5 else "Negative 😞"
    return label, float(pred)

# -------------------------
# Test sample
# -------------------------
sample_text = "I really hate this movie, it was fantastic!"

baseline_pred, baseline_score = predict_sentiment(baseline_model, sample_text, tokenizer)
print("Baseline Model Prediction:", baseline_pred, "(score:", baseline_score, ")")
