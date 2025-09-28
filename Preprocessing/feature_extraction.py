import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_imdb(num_words=10000, maxlen=200):
    """
    Load and preprocess the IMDB sentiment dataset.
    
    Args:
        num_words (int): Keep only the top 'num_words' most frequent words.
        maxlen (int): Maximum review length after padding/truncating.
    
    Returns:
        X_train, y_train, X_test, y_test: Preprocessed train/test splits.
    """
    # -------------------------------
    # 1. Load IMDB dataset
    # -------------------------------
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    print("Training samples:", len(X_train))
    print("Test samples:", len(X_test))
    print("First review (raw indices):", X_train[0][:10])

    # -------------------------------
    # 2. Pad Sequences
    # -------------------------------
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')

    print("Shape of X_train:", X_train.shape)
    print("Shape of X_test:", X_test.shape)

    # -------------------------------
    # 3. Convert labels
    # -------------------------------
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    print("Unique labels:", np.unique(y_train))

    return X_train, y_train, X_test, y_test
