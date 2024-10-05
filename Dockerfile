# Stage 1: Data Processing
FROM python:3.11-slim as processor

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for processing
COPY requirements.txt ./
COPY src/process.py ./src/
COPY data/yelp_academic_dataset_review.json ./data/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-binary :all: wordcloud

# Run the processing script
RUN python src/process.py

# Stage 2: Training
FROM python:3.11-slim as trainer

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for training
COPY requirements.txt ./
COPY src/train.py ./src/
COPY --from=processor /app/yelp_reviews_train.csv /app/yelp_reviews_test.csv ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-binary :all: wordcloud

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords wordnet

# Set Neptune.ai credentials as build arguments
ARG NEPTUNE_API_TOKEN
ARG NEPTUNE_PROJECT

# Set environment variables
ENV NEPTUNE_API_TOKEN=${NEPTUNE_API_TOKEN}
ENV NEPTUNE_PROJECT=${NEPTUNE_PROJECT}

# Run the training script
RUN python src/train.py

# Stage 3: Deployment
FROM python:3.11-slim

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only the necessary files for deployment
COPY requirements.txt ./
COPY src/app.py src/model_deployment.py ./src/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir --no-binary :all: wordcloud

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Copy the trained model from the previous stage
COPY --from=trainer /app/sentiment_model.joblib ./
COPY --from=trainer /app/vectorizer.pkl ./

# Set environment variables for Flask
ENV FLASK_APP=src/app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=8080

# Expose the port the app runs on
EXPOSE 8080

# Run the Flask application
CMD ["flask", "run"]