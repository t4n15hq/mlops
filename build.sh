#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Check if Neptune credentials are provided
if [ -z "$NEPTUNE_API_TOKEN" ] || [ -z "$NEPTUNE_PROJECT" ]; then
    echo "Error: Please set NEPTUNE_API_TOKEN and NEPTUNE_PROJECT environment variables"
    exit 1
fi

# Check if the Yelp dataset exists
YELP_DATASET="data/yelp_academic_dataset_review.json"
if [ ! -f "$YELP_DATASET" ]; then
    echo "Error: Please ensure '$YELP_DATASET' exists"
    exit 1
fi

# Build the Docker image
echo "Building Docker image..."
docker build --build-arg NEPTUNE_API_TOKEN=$NEPTUNE_API_TOKEN \
             --build-arg NEPTUNE_PROJECT=$NEPTUNE_PROJECT \
             -t sentiment-analysis-flask .

# Check if build was successful
if [ $? -eq 0 ]; then
    echo "Docker image built successfully"
else
    echo "Error: Docker build failed"
    exit 1
fi

# Run the Docker container
echo "Running Docker container..."
docker run -p 8080:8080 sentiment-analysis-flask

# Optionally, you can add a trap to handle Ctrl+C and stop the container gracefully
trap 'echo "Stopping container..."; docker stop $(docker ps -q --filter ancestor=sentiment-analysis-flask)' SIGINT