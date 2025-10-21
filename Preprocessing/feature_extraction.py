# import pandas as pd
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

# def load_and_preprocess_imdb(csv_path="data/cleaned_dataset.csv", 
#                                     num_words=10000, maxlen=200, test_size=0.2, random_state=42):
#     """
#     Load your custom CSV dataset, tokenize and pad sequences, and return
#     X_train, y_train, X_test, y_test, tokenizer.
    
#     CSV must have 'review' and 'sentiment' columns. Sentiment should be 0/1 or 'positive'/'negative'.
#     """
#     print("🔹 Loading custom CSV dataset...")
#     df = pd.read_csv(csv_path)
    
#     if 'review' not in df.columns or 'sentiment' not in df.columns:
#         raise ValueError("CSV must contain 'review' and 'sentiment' columns")

#     # Convert sentiment to 0/1 if it's text
#     if df['sentiment'].dtype == object:
#         df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

#     texts = df['review'].tolist()
#     labels = df['sentiment'].values

#     # Split into train/test
#     X_train_texts, X_test_texts, y_train, y_test = train_test_split(
#         texts, labels, test_size=test_size, random_state=random_state
#     )

#     print("🔹 Tokenizing texts...")
#     tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
#     tokenizer.fit_on_texts(X_train_texts)

#     X_train = tokenizer.texts_to_sequences(X_train_texts)
#     X_test = tokenizer.texts_to_sequences(X_test_texts)

#     print("🔹 Padding sequences...")
#     X_train = pad_sequences(X_train, maxlen=maxlen)
#     X_test = pad_sequences(X_test, maxlen=maxlen)

#     print("✅ Custom dataset preprocessing complete.")
#     return X_train, y_train, X_test, y_test, tokenizer

# # Example usage
# # X_train, y_train, X_test, y_test, tokenizer = load_and_preprocess_custom_imdb()

# This is Newer by Chatgpt 
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Download NLTK resources (run once)
nltk.download('stopwords')
nltk.download('wordnet')

def load_and_preprocess_imdb(csv_path="data/cleaned_dataset.csv", 
                             num_words=10000, maxlen=200, test_size=0.2, random_state=42,
                             save_tokenizer_path="results/tokenizer.pkl"):
    """
    Load CSV dataset, clean text, convert labels, tokenize, pad sequences, and save tokenizer.
    Returns: X_train, y_train, X_test, y_test, tokenizer
    """
    print("This is Latest Feature Extraction ")
    print("🔹 Loading dataset...")
    df = pd.read_csv(csv_path)

    # Ensure required columns exist
    if 'review' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'review' and 'sentiment' columns")

    # Map sentiment to 0/1 if text
    if df['sentiment'].dtype == object:
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Fill any remaining NaNs in sentiment
    df['sentiment'] = df['sentiment'].fillna(0)

    # Stopwords and negation handling
    all_stopwords = stopwords.words('english')
    negation = ['no', 'not']
    all_stopwords = [w for w in all_stopwords if w not in negation]

    lemma = WordNetLemmatizer()
    
    # Text cleaning function
    def clean_text(text):
        text = str(text)
        text = re.sub(r"http\S+", " ", text)          # remove URLs
        text = re.sub(r"<.*?>", " ", text)           # remove HTML tags
        text = re.sub(r"&\w+", " ", text)            # remove &codes
        text = re.sub(r"@\w+", " ", text)            # remove mentions
        text = re.sub(r"#\w+", " ", text)            # remove hashtags
        text = re.sub(r"[^a-zA-Z]", " ", text)       # keep only letters
        words = text.lower().split()
        words = [lemma.lemmatize(word) for word in words if word not in all_stopwords]
        return " ".join(words)
    
    print("🔹 Cleaning reviews...")
    df['review'] = df['review'].apply(clean_text)
    
    texts = df['review'].tolist()
    labels = df['sentiment'].values

    # Train-test split
    X_train_texts, X_test_texts, y_train, y_test = train_test_split(
        texts, labels, test_size=test_size, random_state=random_state
    )

    # Tokenizer
    print("🔹 Tokenizing texts...")
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_texts)

    X_train = tokenizer.texts_to_sequences(X_train_texts)
    X_test = tokenizer.texts_to_sequences(X_test_texts)

    print("🔹 Padding sequences...")
    X_train = pad_sequences(X_train, maxlen=maxlen)
    X_test = pad_sequences(X_test, maxlen=maxlen)

    # Save tokenizer
    with open(save_tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"✅ Tokenizer saved to {save_tokenizer_path}")

    print("✅ Preprocessing complete!")
    return X_train, y_train, X_test, y_test, tokenizer
