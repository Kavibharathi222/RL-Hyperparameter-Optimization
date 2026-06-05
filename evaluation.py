import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------
# Load model + tokenizer
# -------------------------

# Load baseline model
baseline_model = tf.keras.models.load_model("models/saved_models/baseline_model.h5")

# Load RL-trained model
rl_model = tf.keras.models.load_model("results/rl_trained_model.h5")

# Load tokenizer
with open("results/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# Helper function
# -------------------------
def predict_sentiment(model, text, tokenizer, maxlen=200):
    # Convert text -> sequence
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    
    # Predict
    pred = model.predict(padded, verbose=0)[0][0]
    
    # Map to label
    label = "Positive 😀" if pred >= 0.5 else "Negative 😞"
    return label, float(pred)

# -------------------------
# Test your models
# -------------------------
sample_text = "I really loved this movie, it was fantastic!"

baseline_pred, baseline_score = predict_sentiment(baseline_model, sample_text, tokenizer)
rl_pred, rl_score = predict_sentiment(rl_model, sample_text, tokenizer)

print("Baseline Model Prediction:", baseline_pred, "(score:", baseline_score, ")")
print("RL-Trained Model Prediction:", rl_pred, "(score:", rl_score, ")")
