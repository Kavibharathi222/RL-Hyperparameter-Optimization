from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

def build_baseline_model(input_dim=10000, embedding_dim=128, maxlen=200):
    """
    Baseline BiLSTM model for Sentiment Analysis.
    """
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen),
        Bidirectional(LSTM(128, return_sequences=False)),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
