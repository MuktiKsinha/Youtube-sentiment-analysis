import os
import pickle
import logging
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import yaml

# ---------------------------
# Logging configuration
# ---------------------------
logger = logging.getLogger('Feature_Engineering')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('Feature_Engineering_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------
# Helper functions
# ---------------------------
def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def path_interim(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/interim', filename)

def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)

def load_params(param_path: str) -> dict:
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', param_path)
        return params
    except FileNotFoundError:
        logger.error('File not found %s', param_path)
        raise
    except yaml.YAMLError as e:
        logger.error('Yaml error %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error %s', e)
        raise

def load_data(file_path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(file_path)
        df.fillna('', inplace=True)  # Fill NaN values
        logger.debug('Data loaded from %s', file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse CSV file: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading data: %s', e)
        raise

# ---------------------------
# BOW function
# ---------------------------
def apply_bow(train_data: pd.DataFrame, max_features: int, ngram_range: tuple):
    """Apply Bag-of-Words to training data."""
    try:
        vectorizer = CountVectorizer(ngram_range=ngram_range, max_features=max_features)
        X_train = train_data['clean_comment'].values
        y_train = train_data['category'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        logger.debug(f'BOW transformation complete. Train shape: {X_train.shape}')

        # Save the vectorizer
        with open(path_processed('bow_vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)

        return X_train_bow, y_train
    except Exception as e:
        logger.error('Error during BOW transformation: %s', e)
        raise

# -----------------------
# Main pipeline
# -----------------------
def main():
    try:
        logger.info("Starting Feature Engineering...")

        root_dir = get_root_directory()
        params = load_params(os.path.join(root_dir, 'params.yaml'))
        max_features = params['feature_engineering']['max_features']
        ngram_range = tuple(params['feature_engineering']['ngram_range'])

        # Load preprocessed training data
        train_data = load_data(os.path.join(root_dir, 'data/interim/train_processed.csv'))

        # Apply BOW only
        X_train_bow, y_train = apply_bow(train_data, max_features, ngram_range)

        # Save final training data
        with open(path_processed("X_train_BOW.pkl"), "wb") as f:
            pickle.dump(X_train_bow, f)

        with open(path_processed("y_train.pkl"), "wb") as f:
            pickle.dump(y_train, f)

        logger.info("Feature Engineering Completed Successfully (BOW only)")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        raise

if __name__ == "__main__":
    main()

    






