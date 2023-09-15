import streamlit as st
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.preprocessing.text import Tokenizer, tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
from keras.models import load_model

# Initialize global variables
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
max_length = 100

# Load the saved model
loaded_model = load_model("sentiment.h5")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_stopwords(text):
    words = word_tokenize(text)
    filtered = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered)

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

def preprocess_user_input(text):
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)

    # Loading the tokenizer vocabulary
    with open('tokenizer_vocabulary.json', 'r', encoding='utf-8') as f:
        data = f.read()
        tokenizer = tokenizer_from_json(data)

    sequences = tokenizer.texts_to_sequences([text])
    
    # Check if the sequences are empty
    if not sequences or not sequences[0]:
        return None
        
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequences

def display_help_menu():
    st.subheader("1. What is this application?")
    st.write("""
    This application analyzes the sentiment of any given text. 
    Just type in your text and the app will determine whether it's positive or negative.
    """)

    st.subheader("2. How to use?")
    st.write("""
    - Enter your text in the text area.
    - Click on 'Predict' to get the sentiment.
    """)

    st.subheader("3. Troubleshooting")
    st.write("""
    If you encounter any issues:
    - Try rephrasing your text.
    - Ensure your text is in English.
    - Check the format of your text.
    """)

# Streamlit UI
st.title("SentimAnalyzer")
st.sidebar.header("Menu")
help_option = st.sidebar.radio('Choose an option', ['Home', 'Help'])

if help_option == 'Home':
    user_text = st.text_area("Enter text for sentiment analysis:")

    if st.button('Predict'):
        processed_input = preprocess_user_input(user_text)
        
        # Check if the processed_input is None and inform the user
        if processed_input is None:
            st.write("Unable to process the input text. Please try another.")
        else:
            prediction = loaded_model.predict(processed_input)
            sentiment = "Positive" if np.argmax(prediction) > 0.5 else "Negative"
            st.write(f"The text sentiment is: {sentiment}")

elif help_option == 'Help':
    display_help_menu()
