import joblib
import pickle
import os
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
import ssl
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(current_dir)

# Ensure SSL context is unverified (to avoid issues with SSL certs)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download necessary NLTK resources if they are not already downloaded
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    logging.error(f"Error downloading NLTK resources: {e}")
    raise

stop_words = set(stopwords.words('english'))

def load_model_and_vectorizer():
    model_path = os.path.join(project_root, 'sentiment_model.joblib')
    vectorizer_path = os.path.join(project_root, 'vectorizer.pkl')
    
    logging.info(f"Attempting to load model from: {model_path}")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            logging.info(f"Successfully loaded model from {model_path}")
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            raise
    else:
        logging.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logging.info(f"Attempting to load vectorizer from: {vectorizer_path}")
    if os.path.exists(vectorizer_path):
        try:
            with open(vectorizer_path, 'rb') as f:
                vectorizer = pickle.load(f)
            # Ensure the vectorizer is fitted
            if not hasattr(vectorizer, 'idf_'):
                raise ValueError("Loaded vectorizer is not fitted.")
            logging.info("Successfully loaded vectorizer with idf_ attribute")
        except Exception as e:
            logging.error(f"Error loading vectorizer: {e}")
            raise
    else:
        logging.error(f"Vectorizer file not found: {vectorizer_path}")
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
    
    return model, vectorizer

def ensure_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')

def preprocess_text(text):
    ensure_nltk_data()
    tokens = word_tokenize(text.lower())
    return ' '.join([token for token in tokens if token.isalnum() and token not in stop_words])

def predict_sentiment(text, model, vectorizer):
    try:
        processed_text = preprocess_text(text)
        vectorized_text = vectorizer.transform([processed_text])
        prediction = model.predict(vectorized_text)[0]
        return "positive" if prediction == 1 else "negative"
    except Exception as e:
        logging.error(f"Error in predict_sentiment: {e}")
        raise

if __name__ == "__main__":
    try:
        model, vectorizer = load_model_and_vectorizer()
        logging.info("Model and vectorizer loaded successfully.")
        
        # Test the model
        sample_text = "This product is amazing! I love it."
        result = predict_sentiment(sample_text, model, vectorizer)
        print(f"The sentiment of '{sample_text}' is: {result}")
    except Exception as e:
        logging.error(f"An error occurred: {e}")