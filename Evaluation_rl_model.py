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
sample_texts = [
    "One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. They are right, as this is exactly what happened with me.<br /><br />The first thing that struck me about Oz was its brutality and unflinching scenes of violence, which set in right from the word GO. Trust me, this is not a show for the faint hearted or timid. This show pulls no punches with regards to drugs, sex or violence. Its is hardcore, in the classic use of the word.<br /><br />It is called OZ as that is the nickname given to the Oswald Maximum Security State Penitentary. It focuses mainly on Emerald City, an experimental section of the prison where all the cells have glass fronts and face inwards, so privacy is not high on the agenda. Em City is home to many..Aryans, Muslims, gangstas, Latinos, Christians, Italians, Irish and more....so scuffles, death stares, dodgy dealings and shady agreements are never far away.<br /><br />I would say the main appeal of the show is due to the fact that it goes where other shows wouldn't dare. Forget pretty pictures painted for mainstream audiences, forget charm, forget romance...OZ doesn't mess around. The first episode I ever saw struck me as so nasty it was surreal, I couldn't say I was ready for it, but as I watched more, I developed a taste for Oz, and got accustomed to the high levels of graphic violence. Not just violence, but injustice (crooked guards who'll be sold out for a nickel, inmates who'll kill on order and get away with it, well mannered, middle class inmates being turned into prison bitches due to their lack of street skills or prison experience) Watching Oz, you may become comfortable with what is uncomfortable viewing....thats if you can get in touch with your darker side.",
    """I can see what they were trying to pull off here, and they almost did it. Emma Paunil , and Brianna Roy don't have a lot of experience between them, but there is potential for both of their careers. This venture however fell just a little short of being a complete effort though. Mostly it was the sound that had me cringing. Up until the party scene, it was horrible. There was an tinny sound happening throughout until then. I don't know why the sound engineers didn't clue into it until that party scene.The zany, offbeat dialogue, and story line kept me entertained enough to get through the sound issues. It fell off the rails a bit during the party scene. Was it too much to ask for the solo cups to at least appear filled with any sort of beverage? Aside from establishing an alibi, the whole scene felt disjointed.

Overall it was a good venture for a crew with limited experienc"""
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
