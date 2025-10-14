import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF C++ logs

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import logging
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import csv

# -------------------------
# Load baseline model + tokenizer
# -------------------------
baseline_model = tf.keras.models.load_model("SavedModels/baseline_model.keras")
print("Baseline model input shape:", baseline_model.input_shape)

with open("SavedModels/tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# -------------------------
# Load optimal threshold from training metrics
# -------------------------
base_threshold = 0.10  # default fallback
try:
    with open("results/test_metrics.csv", "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "Best_Threshold" in row:
                base_threshold = float(row["Best_Threshold"])
    print(f"✅ Loaded optimal threshold: {base_threshold}")
except Exception as e:
    print(f"⚠️ Could not load threshold, using default {base_threshold}. Error: {e}")

# -------------------------
# Lexicon for short reviews
# -------------------------
positive_words = {"good", "great", "excellent", "enjoy", "love", "amazing", "fantastic", "awesome"}
negative_words = {"bad", "terrible", "worst", "hate", "boring", "awful", "poor", "crap"}

def lexicon_sentiment(text):
    """
    Simple lexicon-based sentiment for short reviews
    """
    text_lower = text.lower()
    pos_score = sum(word in text_lower for word in positive_words)
    neg_score = sum(word in text_lower for word in negative_words)
    
    if pos_score > neg_score:
        return "Positive 😀"
    elif neg_score > pos_score:
        return "Negative 😞"
    else:
        return "Neutral 😐"  # fallback if unclear

# -------------------------
# Helper function
# -------------------------
def predict_sentiment_hybrid(model, text, tokenizer, maxlen=200, threshold=base_threshold, short_len=10):
    words = text.split()
    if len(words) <= short_len:
        # Use lexicon for short reviews
        label = lexicon_sentiment(text)
        return label, None, "lexicon"
    else:
        # Use BiLSTM for long reviews
        seq = tokenizer.texts_to_sequences([text])
        padded = pad_sequences(seq, maxlen=maxlen, padding="post", truncating="post")
        pred_prob = model.predict(padded, verbose=0)[0][0]
        label = "Positive 😀" if pred_prob < threshold else "Negative 😞"
        return label, float(pred_prob), threshold
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

# -------------------------
# Run predictions
# -------------------------
for text in sample_texts:
    print("\nReview:", text)
    token_seq = tokenizer.texts_to_sequences([text])
    print("Tokenized:", token_seq)
    pred_label, pred_score, method = predict_sentiment_hybrid(baseline_model, text, tokenizer)
    if method == "lexicon":
        print(f"Prediction (lexicon): {pred_label}")
    else:
        print(f"Prediction (BiLSTM): {pred_label} (score: {pred_score:.4f}, threshold used: {method})")
