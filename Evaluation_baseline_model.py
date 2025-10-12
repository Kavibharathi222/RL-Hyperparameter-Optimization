import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

# -------------------------
# Load baseline model + tokenizer
# -------------------------
baseline_model = tf.keras.models.load_model("SavedModels/baseline_model.keras")
print("Baseline model input shape:", baseline_model.input_shape)

with open("SavedModels/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# Load optimal threshold from training metrics
# -------------------------
base_threshold = 0.10  # default fallback
try:
    with open("results/test_metrics.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Best_Threshold" in row:
                base_threshold = float(row["Best_Threshold"])
    print(f"✅ Loaded optimal threshold: {base_threshold}")
except Exception as e:
    print(f"⚠️ Could not load threshold, using default {base_threshold}. Error: {e}")

# -------------------------
# Lexicon for short reviews
# -------------------------
positive_words = {"good", "great", "excellent", "enjoy", "love", "amazing", "fantastic", "awesome"}
negative_words = {"bad", "terrible", "worst", "hate", "boring", "awful", "poor", "crap"}

def lexicon_sentiment(text):
    """
    Simple lexicon-based sentiment for short reviews
    """
    text_lower = text.lower()
    pos_score = sum(word in text_lower for word in positive_words)
    neg_score = sum(word in text_lower for word in negative_words)
    
    if pos_score > neg_score:
        return "Positive 😀"
    elif neg_score > pos_score:
        return "Negative 😞"
    else:
        return "Neutral 😐"  # fallback if unclear

# -------------------------
# Helper function
# -------------------------
def predict_sentiment_hybrid(model, text, tokenizer, maxlen=200, threshold=base_threshold, short_len=10):
    words = text.split()
    if len(words) <= short_len:
        # Use lexicon for short reviews
        label = lexicon_sentiment(text)
        return label, None, "lexicon"
    else:
        # Use BiLSTM for long reviews
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        pred_prob = model.predict(padded, verbose=0)[0][0]
        label = "Positive 😀" if pred_prob < threshold else "Negative 😞"
        return label, float(pred_prob), threshold

# -------------------------
# Sample reviews
# -------------------------
sample_texts = [
    "Crap, crap and totally crap. Did I mention this film was totally crap? Well, it's totally crap",
    "I Really Hate this Movie The Movie is Very Bad",
    "Praised for his intensity, action, screen presence, and dialogue delivery, with fans celebrating his continued success.",
    "I Really Enjoy This Movie",
    "This is Very Bad Movie",
    "Movie so Boring"
]

# -------------------------
# Run predictions
# -------------------------
for text in sample_texts:
    print("\nReview:", text)
    token_seq = tokenizer.texts_to_sequences([text])
    print("Tokenized:", token_seq)
    pred_label, pred_score, method = predict_sentiment_hybrid(baseline_model, text, tokenizer)
    if method == "lexicon":
        print(f"Prediction (lexicon): {pred_label}")
    else:
        print(f"Prediction (BiLSTM): {pred_label} (score: {pred_score:.4f}, threshold used: {method})")
