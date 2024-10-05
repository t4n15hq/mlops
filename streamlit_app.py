import streamlit as st
from src.model_deployment import load_model_and_vectorizer, predict_sentiment
import os
import logging
import nltk

@st.cache_resource
def download_nltk_data():
    nltk.download('punkt')
    nltk.download('stopwords')

download_nltk_data()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

nltk.download('punkt', quiet=True)  # Download the 'punkt' tokenizer
nltk.download('stopwords', quiet=True)  # Download stopwords
nltk.download('wordnet', quiet=True)  # Download WordNet

# Set page config (move this to the top)
st.set_page_config(page_title="Sentiment Analysis", layout="centered")

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(current_dir)

# Load model and vectorizer at app startup
@st.cache_resource
def load_model():
    try:
        model, vectorizer = load_model_and_vectorizer()
        logging.info("Model and vectorizer loaded successfully")
        return model, vectorizer
    except Exception as e:
        logging.error(f"Error loading model or vectorizer: {e}")
        st.error(f"Error loading model or vectorizer: {e}")
        return None, None

model, vectorizer = load_model()

# Custom CSS
st.markdown("""
<style>
    .stTextInput > div > div > input {
        min-height: 100px;
    }
    .stButton > button {
        width: 100%;
        background-color: #3498db;
        color: white;
    }
    .stButton > button:hover {
        background-color: #2980b9;
    }
    .positive {
        color: #27ae60;
        font-weight: bold;
    }
    .negative {
        color: #c0392b;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.title("Sentiment Analysis")

# Text input
text = st.text_area("Enter your text here...", height=150)

# Predict button
if st.button("Analyze Sentiment"):
    if text:
        try:
            if model is None or vectorizer is None:
                st.error("Model or vectorizer not loaded properly")
            else:
                sentiment = predict_sentiment(text, model, vectorizer)
                sentiment_class = "positive" if sentiment == "positive" else "negative"
                st.markdown(f"Sentiment: <span class='{sentiment_class}'>{sentiment}</span>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {e}")
            logging.error(f"Error during prediction: {e}")
    else:
        st.warning("Please enter some text to analyze.")

# Optional: Add some information about the app
st.markdown("---")
st.markdown("This app uses a machine learning model to predict the sentiment of the input text.")