import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import neptune
from neptune.types import File
from neptune.utils import stringify_unsupported
import time
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import gc
import joblib
import os
import ssl
import pickle
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(current_dir)

logging.info(f"Current working directory: {os.getcwd()}")
logging.info(f"Project root: {project_root}")
logging.info(f"Expected data file path: {os.path.join(project_root, 'data', 'yelp_reviews_train.csv')}")
logging.info(f"Does the file exist? {os.path.exists(os.path.join(project_root, 'data', 'yelp_reviews_train.csv'))}")

# Ensure SSL context is unverified (to avoid issues with SSL certs)
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Neptune initialization
run = neptune.init_run(project="t4n15hq/SentimentAnalysis")

# Start timing
start_time = time.time()

# Log project-level metadata
run["project/description"] = "Sentiment Analysis project on Yelp reviews using Logistic Regression"
run["project/data_source"] = "Yelp reviews dataset"
run["project/model_type"] = "Logistic Regression"
run["project/dataset/train"] = os.path.join(project_root, 'data', 'yelp_reviews_train.csv')
run["project/dataset/test"] = os.path.join(project_root, 'data', 'yelp_reviews_test.csv')
run["project/dataset/current_version"] = "v1.0"

# Download necessary NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# Text preprocessing
logging.info("Preparing preprocessing...")
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens if token.isalnum() and token not in stop_words])

# Log preprocessing steps
run["preprocessing/steps"] = [
    "Lowercase conversion",
    "Tokenization",
    "Stopword removal",
    "Lemmatization"
]

# Generator function to load and preprocess data in chunks
def data_generator(file_path, chunk_size=10000):
    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        chunk['processed_text'] = chunk['text'].apply(preprocess_text)
        yield chunk[['processed_text', 'sentiment']]

# Load and preprocess data in chunks
logging.info("Loading and preprocessing data...")
train_data = pd.concat(data_generator(os.path.join(project_root, 'data', 'yelp_reviews_train.csv')))
test_data = pd.concat(data_generator(os.path.join(project_root, 'data', 'yelp_reviews_test.csv')))

# Log dataset info
run["data/train_shape"] = str(train_data.shape)
run["data/test_shape"] = str(test_data.shape)
run["data/train_classes"] = stringify_unsupported(train_data['sentiment'].value_counts().to_dict())
run["data/test_classes"] = stringify_unsupported(test_data['sentiment'].value_counts().to_dict())

# TF-IDF Vectorization
logging.info("Vectorizing text...")
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train = vectorizer.fit_transform(train_data['processed_text'])
X_test = vectorizer.transform(test_data['processed_text'])

# Save the fitted vectorizer
logging.info("Saving vectorizer...")
with open(os.path.join(project_root, "vectorizer.pkl"), 'wb') as f:
    pickle.dump(vectorizer, f)

# Verify the saved vectorizer
with open(os.path.join(project_root, "vectorizer.pkl"), 'rb') as f:
    loaded_vectorizer = pickle.load(f)
if hasattr(loaded_vectorizer, 'idf_'):
    logging.info("Vectorizer saved and loaded successfully with idf_ attribute.")
else:
    logging.error("Saved vectorizer does not have idf_ attribute.")

# Check if vectorizer is fitted
if not hasattr(vectorizer, 'idf_'):
    raise ValueError("Vectorizer is not fitted. Check the fitting process.")

# Log vectorizer info
run["preprocessing/vectorizer"] = "TF-IDF"
run["preprocessing/max_features"] = 5000

# Train model
logging.info("Training model...")
model_clf = LogisticRegression(C=1.0, random_state=42)
model_clf.fit(X_train, train_data['sentiment'])

# Log model parameters
run["model/parameters"] = {
    "vectorizer__max_features": 5000,
    "vectorizer__ngram_range": (1, 2),
    "classifier__C": 1.0,
    "classifier__random_state": 42
}

# Evaluate model
logging.info("Evaluating model...")
y_pred = model_clf.predict(X_test)
y_prob = model_clf.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(test_data['sentiment'], y_pred)
classification_rep = classification_report(test_data['sentiment'], y_pred, output_dict=True)
conf_matrix = confusion_matrix(test_data['sentiment'], y_pred)
roc_auc = roc_auc_score(test_data['sentiment'], y_prob)

# Log metrics
run["metrics/accuracy"] = float(accuracy)
run["metrics/classification_report"] = stringify_unsupported(classification_rep)
run["metrics/confusion_matrix"] = stringify_unsupported(conf_matrix.tolist())
run["metrics/roc_auc"] = float(roc_auc)

# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(test_data['sentiment'], y_prob)
run["metrics/precision"] = stringify_unsupported(precision.tolist())
run["metrics/recall"] = stringify_unsupported(recall.tolist())

# Feature importance
feature_importance = pd.DataFrame({
    'feature': vectorizer.get_feature_names_out(),
    'importance': model_clf.coef_[0]
}).sort_values('importance', ascending=False)

run["model/top_positive_features"] = stringify_unsupported(feature_importance.head(20).to_dict())
run["model/top_negative_features"] = stringify_unsupported(feature_importance.tail(20).to_dict())

# Generate and log visualizations
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.savefig(os.path.join(project_root, 'confusion_matrix.png'))
run["visualizations/confusion_matrix"].upload(os.path.join(project_root, 'confusion_matrix.png'))
plt.close()

plt.figure(figsize=(10, 8))
plt.plot(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.savefig(os.path.join(project_root, 'precision_recall_curve.png'))
run["visualizations/precision_recall_curve"].upload(os.path.join(project_root, 'precision_recall_curve.png'))
plt.close()

# Word clouds
positive_words = ' '.join(feature_importance.head(100)['feature'])
negative_words = ' '.join(feature_importance.tail(100)['feature'])

wordcloud_positive = WordCloud(width=800, height=400, background_color='white').generate(positive_words)
wordcloud_negative = WordCloud(width=800, height=400, background_color='white').generate(negative_words)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_positive, interpolation='bilinear')
plt.axis('off')
plt.title('Top Positive Words')
plt.savefig(os.path.join(project_root, 'wordcloud_positive.png'))
run["visualizations/wordcloud_positive"].upload(os.path.join(project_root, 'wordcloud_positive.png'))
plt.close()

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud_negative, interpolation='bilinear')
plt.axis('off')
plt.title('Top Negative Words')
plt.savefig(os.path.join(project_root, 'wordcloud_negative.png'))
run["visualizations/wordcloud_negative"].upload(os.path.join(project_root, 'wordcloud_negative.png'))
plt.close()

# Log example predictions
sample_size = min(20, len(test_data))
sample_indices = np.random.choice(len(test_data), sample_size, replace=False)
for i, idx in enumerate(sample_indices):
    run[f"predictions/example_{i}/text"] = test_data['processed_text'].iloc[idx]
    run[f"predictions/example_{i}/true_sentiment"] = int(test_data['sentiment'].iloc[idx])
    run[f"predictions/example_{i}/predicted_sentiment"] = int(y_pred[idx])
    run[f"predictions/example_{i}/prediction_probability"] = float(y_prob[idx])

# Save and log the model
logging.info("Saving model and vectorizer...")
joblib.dump(model_clf, os.path.join(project_root, "sentiment_model.joblib"))
with open(os.path.join(project_root, "vectorizer.pkl"), 'wb') as f:
    pickle.dump(vectorizer, f)

logging.info("Model and vectorizer saved successfully.")

# Verify saved vectorizer
loaded_vectorizer = joblib.load(os.path.join(project_root, "vectorizer.pkl"))
if hasattr(loaded_vectorizer, 'idf_'):
    logging.info("Saved vectorizer is properly fitted.")
else:
    logging.error("Saved vectorizer is not fitted. Check the saving process.")

# Log execution time
end_time = time.time()
run["execution/total_time_seconds"] = float(end_time - start_time)

# Stop the Neptune run
run.stop()

logging.info("Training and logging completed!")

# Clean up
del train_data, test_data, X_train, X_test, y_pred, y_prob
gc.collect()