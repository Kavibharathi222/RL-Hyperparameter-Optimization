import os
import pickle
import csv
import warnings
import logging
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Preprocessing.feature_extraction import load_and_preprocess_imdb

# -------------------------
# Suppress TF warnings
# -------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -------------------------
# Load RL model and tokenizer
# -------------------------
rl_model = load_model("SavedModels/final_model.keras")
with open("SavedModels/tokenizer_basemodel.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("✅ RL model and tokenizer loaded successfully.")

# -------------------------
# Prediction function
# -------------------------
def predict_sentiment(model, tokenizer, text, maxlen=200, threshold=0.3):
    """
    Preprocesses text, predicts sentiment, and applies threshold.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred_prob = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if pred_prob >= threshold else "Negative 😞"
    return label, float(pred_prob)

# -------------------------
# Evaluate sample reviews
# -------------------------
sample_texts = [
    "Crap, crap and totally crap. Did I mention this film was totally crap? Well, it's totally crap",
    "I Really Hate this Movie The Movie is Very Bad",
    "Praised for his intensity, action, screen presence, and dialogue delivery, with fans celebrating his continued success.",
    "I Really Enjoy This Movie",
    "This is Very Bad Movie",
    "Movie so Boring"
]

threshold = 0.3
results_file = "results/rl_review_predictions.csv"
os.makedirs("results", exist_ok=True)

with open(results_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Review", "Prediction", "Probability", "Threshold"])
    for text in sample_texts:
        label, score = predict_sentiment(rl_model, tokenizer, text, threshold=threshold)
        print(f"Review: {text}")
        print(f"Prediction: {label} (score: {score:.4f}, threshold: {threshold})\n")
        writer.writerow([text, label, score, threshold])

print(f"✅ Predictions saved to {results_file}")

# -------------------------
# Evaluate on full test dataset
# -------------------------
print("\n🔹 Loading full IMDB test dataset...")
_, _, X_test, y_test, _ = load_and_preprocess_imdb(num_words=10000, maxlen=200)

print("🔹 Padding test sequences...")
X_test_padded = pad_sequences(X_test, maxlen=200, padding="post", truncating="post")
print("✅ Dataset preprocessing complete.")

# Predict
y_pred_probs = rl_model.predict(X_test_padded, verbose=0)
y_pred = (y_pred_probs >= threshold).astype(int)

# Compute metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Test Metrics (using threshold {:.2f}):".format(threshold))
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save test metrics to CSV
# metrics_file = "results/rl_test_metrics.csv"
# with open(metrics_file, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Accuracy", "Precision", "Recall", "F1", "Threshold"])
#     writer.writerow([acc, prec, rec, f1, threshold])

# print(f"✅ Test metrics saved to {metrics_file}")








import os
import pickle
import csv
import warnings
import logging
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Preprocessing.feature_extraction import load_and_preprocess_imdb

# -------------------------
# Suppress TF warnings
# -------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings("ignore", category=FutureWarning)
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# -------------------------
# Load RL model and tokenizer
# -------------------------
rl_model = load_model("SavedModels/final_model.keras")
with open("SavedModels/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

print("✅ RL model and tokenizer loaded successfully.")

# -------------------------
# Prediction function
# -------------------------
def predict_sentiment(model, tokenizer, text, maxlen=200, threshold=0.3):
    """
    Preprocesses text, predicts sentiment, and applies threshold.
    """
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
    pred_prob = model.predict(padded, verbose=0)[0][0]
    label = "Positive 😀" if pred_prob >= threshold else "Negative 😞"
    return label, float(pred_prob)

# -------------------------
# Evaluate sample reviews
# -------------------------
print("First 3 Positive Reviews")
print("Last 3 Negative")
# -------------------------
# Sample reviews
# -------------------------
sample_texts = [
    """I haven't watched many Tamil movies, but I found that Chandramukhi was one of the best yet I watched. It has got everything you can have in a movie: humor, horror, thrill,romance, action, music, etc. The songs are REALLY good in this movie. I would definitely recommend buying the soundtrack of Chandramukhi. There's even one song in Telugu. Overall, I liked this movie a lot. Of course it had Rajnikanth in it, which brought quite a lot more appeal to it. There are good fight scenes with attempts of the "Matrix" effect, which is pretty cool. In the Telugu dub of this movie, one of the songs is sung by Carnatic Diva, Nityasree Mahadevan. What more could one ask for? It is a great movie, and I think everyone should see it. It might be a bit scary for young children, but it should be suitable for older ones. Conclusively, Chandramukhi is a great watch, and with stars like R
    ajnikanth and Jyothika, it is a big success!!""",
    """Movie starts with typical superstar action sequences and flows through interestingly till the end. For audience not aware of movies original version in Malayalam it will be a nice thriller, very unnatural for a Rajini movie.

Rajini has done a great job in mixing all elements. Vadivelu role is commendable. Even though Jyotika has done a reasonable job, her overacting was evident in some scenes. P.Vasu did okay by not screwing up Fazil's work in "Manichitrathalu".

In summary hardcore fans might be disappointed by not hearing any punch lines but nice pace keeps the movie rolling. It's sure to get a lot of appreciation from varied group of audience.""",
    """The storyline is different. No usual generation after generation story. Its a ghost story. Its unlike most of the formula movies that Rajini has made in the recent past. It has his usual mannerisms but lacks punch lines or political dialogs(a welcome change).

It is a ghost movie about an ancestral castle which has the ghost of a dancer who was sodomized and killed by the king who lived there. Vadivelu's comedy plays an integral part in making the movie good.

For a change Rajini does not dominate the screen. In fact its jyotika who dominates the second half of the movie and takes the movie forward. Good fun to watch with a huge gang of friends""",
    """Worst Remake of Malayalam movie Manichitrathazhu. Not only Manichitrathazhu has won two National Film Awards; Best Popular Film Providing Wholesome Entertainment for Fazil and Best Actress for Shobana.""",
    """I happened to see this movie. This movie is as bad as baba. I still don't know why rajni has accepted this movie. It doesn't have any flavor of rajni. 2-fight sequence has been unnecessarily included just to satisfy rajni fans and that's it. Jothika was okay and I couldn't watch Prabhu at all. I have seen the original Manichithrathazhu and nobody in this movie have ever come near to any cast in Malayalam. Certain scenes (especially the end when rajni comes with dog and acts as the vettayapura raja) were height of stupidity. In Malayalam entire movie was great and only this P.Vasu can give such an idiotic version of a beautiful movie. I would advice not to watch this movie. Anyway rajni fans will definitely watch to see what is there. So no point in advising.

I hate this P.Vasu for giving such a stupid movie with our super star.""",
    """I seriously think Rajini should stop acting as the "Super hero" and try to take up some decent roles like father or grandfather....his make-up looks horrible and all the perverted jokes suck. Come on ! u cant even imagine him romancing nayantara ! she is 1/4 of his age.. only solace was good acting by Jo and Prabhu ! Prabhu looks much better in this movie..Jo have over-acted as usual but it didn't look odd as her role demands that.

Last but not the least, all the double meaning dialogues were so bad it no more gives the image that rajini movies are family entertainers...

It looks shameful to appreciate such movies...its nowhere near the original mallu version...

And he shd stop trying to be "keanu reeves" of matrix :-)"""
]


threshold = 0.3
results_file = "results/rl_review_predictions.csv"
os.makedirs("results", exist_ok=True)

with open(results_file, mode="w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Review", "Prediction", "Probability", "Threshold"])
    for text in sample_texts:
        label, score = predict_sentiment(rl_model, tokenizer, text, threshold=threshold)
        print(f"Review: {text}")
        print(f"Prediction: {label} (score: {score:.4f}, threshold: {threshold})\n")
        writer.writerow([text, label, score, threshold])

print(f"✅ Predictions saved to {results_file}")

# -------------------------
# Evaluate on full test dataset
# -------------------------
print("\n🔹 Loading full IMDB test dataset...")
_, _, X_test, y_test, _ = load_and_preprocess_imdb(num_words=10000, maxlen=200)

print("🔹 Padding test sequences...")
X_test_padded = pad_sequences(X_test, maxlen=200, padding="post", truncating="post")
print("✅ Dataset preprocessing complete.")

# Predict
y_pred_probs = rl_model.predict(X_test_padded, verbose=0)
y_pred = (y_pred_probs >= threshold).astype(int)

# Compute metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Test Metrics (using threshold {:.2f}):".format(threshold))
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")

# Save test metrics to CSV
# metrics_file = "results/rl_test_metrics.csv"
# with open(metrics_file, mode="w", newline="") as f:
#     writer = csv.writer(f)
#     writer.writerow(["Accuracy", "Precision", "Recall", "F1", "Threshold"])
#     writer.writerow([acc, prec, rec, f1, threshold])

# print(f"✅ Test metrics saved to {metrics_file}")
