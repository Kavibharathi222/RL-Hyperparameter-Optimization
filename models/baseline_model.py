# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

# def build_baseline_model(input_dim=10000, embedding_dim=128, maxlen=200):
#     """
#     Baseline BiLSTM model for Sentiment Analysis.
#     """
#     model = Sequential([
#         Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen),
#         Bidirectional(LSTM(128, return_sequences=False)),
#         Dropout(0.5),
#         Dense(64, activation='relu'),
#         Dropout(0.5),
#         Dense(1, activation='sigmoid')
#     ])
    
#     model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#     return model

# Update by chatgpt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout

def build_baseline_model(vocab_size=10000, embedding_dim=200, maxlen=200, trainable=False):
    """
    Custom BiLSTM model for Sentiment Analysis based on user preference.
    """
    print("Latest Model is updates ")
    model = Sequential([
        # Embedding layer (can use pre-trained embeddings)
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen, trainable=trainable),
        
        # BiLSTM layer with 200 units
        Bidirectional(LSTM(200, dropout=0.4, recurrent_dropout=0.4)),
        
        # Output layer
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

