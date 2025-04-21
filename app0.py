import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained Keras model
model = load_model('rnn_sentiment_model.h5')

# Define tokenizer settings (should match training)
max_words = 10000
max_len = 100

# You need to refit the tokenizer on the same corpus OR save & load tokenizer
# For demonstration, we'll refit it on a small sample (for real use, save your tokenizer)
# Replace this with a saved tokenizer in production
sample_reviews = ["great coffee", "bad service", "average taste", "loved it", "would not recommend"]
tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
tokenizer.fit_on_texts(sample_reviews)

# Sentiment Labels
labels = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Streamlit UI
st.set_page_config(page_title="☕ Coffee Review Sentiment", layout="centered")
st.title("☕ Coffee Review Sentiment Analyzer")
st.write("Enter a coffee review below, and the RNN model will predict its sentiment.")

review = st.text_area("📝 Your Review", height=150)

if st.button("Analyze Sentiment"):
    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        sequence = tokenizer.texts_to_sequences([review])
        padded = pad_sequences(sequence, maxlen=max_len)
        prediction = model.predict(padded)
        sentiment_class = np.argmax(prediction)
        confidence = np.max(prediction)

        st.subheader("🔍 Prediction:")
        st.write(f"**Sentiment:** {labels[sentiment_class]}")
        st.write(f"**Confidence:** {confidence:.2f}")
