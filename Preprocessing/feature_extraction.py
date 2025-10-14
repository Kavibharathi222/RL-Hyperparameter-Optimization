import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_and_preprocess_imdb(csv_path="data/cleaned_dataset.csv", 
                                    num_words=10000, maxlen=200, test_size=0.2, random_state=42):
    """
    Load your custom CSV dataset, tokenize and pad sequences, and return
    X_train, y_train, X_test, y_test, tokenizer.
    
    CSV must have 'review' and 'sentiment' columns. Sentiment should be 0/1 or 'positive'/'negative'.
    """
    print("🔹 Loading custom CSV dataset...")
    df = pd.read_csv(csv_path)
    
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns")

    # Convert sentiment to 0/1 if it's text
    if df['sentiment'].dtype == object:
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    texts = df['review'].tolist()
    labels = df['sentiment'].values

    # Split into train/test
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    print("🔹 Tokenizing texts...")
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_texts)

    X_train = tokenizer.texts_to_sequences(X_train_texts)
    X_test = tokenizer.texts_to_sequences(X_test_texts)

    print("🔹 Padding sequences...")
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    print("✅ Custom dataset preprocessing complete.")
    return X_train, y_train, X_test, y_test, tokenizer

# Example usage
# X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_custom_imdb()
