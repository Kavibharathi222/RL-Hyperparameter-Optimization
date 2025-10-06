from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


def load_and_preprocess_imdb(num_words=10000, maxlen=200):
    

    print("🔹 Loading IMDB dataset...")
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=num_words)

    print("🔹 Padding sequences...")
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    # Create tokenizer for future manual text testing
    tokenizer = Tokenizer(num_words=num_words)
    word_index = imdb.get_word_index()
    tokenizer.word_index = word_index

    print("✅ Dataset preprocessing complete.")
    # IMPORTANT: return tokenizer here
    return X_train, y_train, X_test, y_test, tokenizer
