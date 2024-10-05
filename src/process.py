import json
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
# Go up one level to the project root
project_root = os.path.dirname(current_dir)

def read_yelp_data(file_path):
    data = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return pd.DataFrame(data)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        raise
    except json.JSONDecodeError:
        logging.error(f"Invalid JSON in file: {file_path}")
        raise

def process_and_save_data():
    input_file = os.path.join(project_root, 'data', 'yelp_academic_dataset_review.json')
    output_train = os.path.join(project_root, 'yelp_reviews_train.csv')
    output_test = os.path.join(project_root, 'yelp_reviews_test.csv')

    logging.info(f"Loading data from {input_file}")
    reviews_df = read_yelp_data(input_file)

    logging.info("Selecting relevant columns and creating sentiment labels")
    reviews_df = reviews_df[['text', 'stars']]
    reviews_df['sentiment'] = (reviews_df['stars'] >= 4).astype(int)
    reviews_df = reviews_df.drop('stars', axis=1)

    logging.info("Balancing the dataset")
    min_sentiment_count = reviews_df['sentiment'].value_counts().min()
    balanced_df = pd.concat([
        reviews_df[reviews_df['sentiment'] == 0].sample(min_sentiment_count, random_state=42),
        reviews_df[reviews_df['sentiment'] == 1].sample(min_sentiment_count, random_state=42)
    ])

    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    logging.info("Splitting into train and test sets")
    train_df, test_df = train_test_split(balanced_df, test_size=0.2, random_state=42)

    logging.info(f"Saving training data to {output_train}")
    train_df.to_csv(output_train, index=False)

    logging.info(f"Saving testing data to {output_test}")
    test_df.to_csv(output_test, index=False)

    logging.info(f"Training set shape: {train_df.shape}")
    logging.info(f"Testing set shape: {test_df.shape}")
    logging.info(f"Sentiment distribution in training set:\n{train_df['sentiment'].value_counts(normalize=True)}")

if __name__ == "__main__":
    try:
        process_and_save_data()
        logging.info("Data processing completed successfully.")
    except Exception as e:
        logging.error(f"An error occurred during data processing: {str(e)}")