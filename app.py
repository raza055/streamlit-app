import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

st.title("IMDB Movie Review Classifier by Ali Raza")

# Cache model
@st.cache_resource
def load_sentiment_model():
    return load_model("simple_rnn_imdb_v2.h5")

model = load_sentiment_model()

# Cache dataset index
@st.cache_resource
def load_index():
    word_index = imdb.get_word_index()
    reverse_word_index = {value: key for key, value in word_index.items()}
    return word_index, reverse_word_index

word_index, reverse_word_index = load_index()

# Preprocess text
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(w, 2) + 3 for w in words]
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

st.write("Enter a movie review for sentiment classification.")
user_input = st.text_area("Movie Review:")

if st.button("Classify"):
    processed = preprocess_text(user_input)
    prediction = model.predict(processed)
    label = "Positive" if prediction[0][0] > 0.5 else "Negative"

    st.write(f"**Sentiment:** {label}")
    st.write(f"**Prediction Score:** {prediction[0][0]:.4f}")
