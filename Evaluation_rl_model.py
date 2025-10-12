import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Hide TF/Keras deprecation warnings

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

# -------------------------
# Rebuild RL model architecture
# -------------------------
# def build_rl_model(vocab_size=10000, maxlen=200, embedding_dim=128):
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     return model


with open("results/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# vocab_size = len(tokenizer.word_index) + 1
# Use the same numbers as training
# vocab_size = 10000
# embedding_dim = 128
# rl_model = build_rl_model(vocab_size=vocab_size, embedding_dim=embedding_dim)
# rl_model.load_weights("SavedModels/final_model.keras")
rl_model = load_model("SavedModels/final_model.keras")
print("RL model summary before loading weights:")
rl_model.summary()
# print("Vocab size in embedding:", vocab_size)



# -------------------------
# Prediction function
# -------------------------
def predict_sentiment(model, tokenizer, text, maxlen=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if pred >= 0.5 else "Negative 😞"
    return label, float(pred)

# -------------------------
# Test sample
# -------------------------
sample_text = "I really hate this movie, it was fantastic!"

rl_pred, rl_score = predict_sentiment(rl_model, tokenizer, sample_text)
print("RL-Trained Model Prediction:", rl_pred, "(score:", rl_score, ")")



import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Hide TF/Keras deprecation warnings

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)


import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import load_model

# -------------------------
# Rebuild RL model architecture
# -------------------------
# def build_rl_model(vocab_size=10000, maxlen=200, embedding_dim=128):
#     model = Sequential()
#     model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
#     model.add(Bidirectional(LSTM(64)))
#     model.add(Dropout(0.5))
#     model.add(Dense(1, activation='sigmoid'))
#     return model


with open("results/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# vocab_size = len(tokenizer.word_index) + 1
# Use the same numbers as training
# vocab_size = 10000
# embedding_dim = 128
# rl_model = build_rl_model(vocab_size=vocab_size, embedding_dim=embedding_dim)
# rl_model.load_weights("SavedModels/final_model.keras")
rl_model = load_model("SavedModels/final_model.keras")
print("RL model summary before loading weights:")
rl_model.summary()
# print("Vocab size in embedding:", vocab_size)



# -------------------------
# Prediction function
# -------------------------
def predict_sentiment(model, tokenizer, text, maxlen=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if pred >= 0.5 else "Negative 😞"
    return label, float(pred)

# -------------------------
# Test sample
# -------------------------
sample_text = "I really hate this movie, it was fantastic!"

rl_pred, rl_score = predict_sentiment(rl_model, tokenizer, sample_text)
print("RL-Trained Model Prediction:", rl_pred, "(score:", rl_score, ")")
