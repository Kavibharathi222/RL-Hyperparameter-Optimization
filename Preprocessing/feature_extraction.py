import numpy as np
from keras.datasets import imdb
from keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_imdb(num_words=10000, maxlen=200):
    """
    Load and preprocess the IMDB dataset.
    """
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    # Pad sequences to ensure equal length
    X_train = pad_sequences(X_train, maxlen=maxlen, padding='post', truncating='post')
    X_test = pad_sequences(X_test, maxlen=maxlen, padding='post', truncating='post')

    return X_train, np.array(y_train), X_test, np.array(y_test)
