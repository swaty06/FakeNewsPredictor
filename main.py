# -*- coding: utf-8 -*-
"""
Created on Wed Apr 16 14:43:20 2025

@author: rampr
"""

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# Load model and tokenizer from Google Drive
model = load_model("sentiment_model.h5")

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

# Parameters
max_sequence_length = 100  # same as used during training

def predict_sentiment(text):
    # Tokenize the text input and pad sequences
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_sequence_length)
    
    # Predict the sentiment using the model (output will be 0 or 1)
    pred = model.predict(padded)
    
    # Return the predicted class (0 = Fake, 1 = Not Fake)
    label = (pred > 0.3).astype(int)  # If the output is greater than 0.5, classify as 1 (Not Fake)
    
    labels = {0: "Fake News", 1: "Real News"}
    
    return labels[label[0][0]]

# Streamlit UI
st.title("📊 Fake News Detection App")
st.write("Enter a sentence to predict if it's fake news:")

user_input = st.text_input("Your sentence:")

if user_input:
    prediction = predict_sentiment(user_input)
    st.success(f"Prediction: **{prediction}**")
