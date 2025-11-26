# src/model/model_building.py

import os
import pickle
import logging
import yaml
import lightgbm as lgb
import numpy as np

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger('model_building')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_building_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------
# Paths
# ---------------------------
def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)

# ---------------------------
# Load parameters
# ---------------------------
def load_params():
    root = get_root_directory()
    with open(os.path.join(root, "params.yaml"), "r") as f:
        params = yaml.safe_load(f)
    return params["model_building"]["lgbm_params"]

# ---------------------------
# Load BOW features
# ---------------------------
def load_features():
    with open(path_processed("X_train_BOW.pkl"), "rb") as f:
        X = pickle.load(f)
    with open(path_processed("y_train.pkl"), "rb") as f:
        y = pickle.load(f)
    X = X.astype('float32')
    logger.debug(f"Loaded features: X={X.shape}, y={y.shape}")
    return X, y

# ---------------------------
# Train and Save model
# ---------------------------
def main():
    try:
        params = load_params()
        X, y = load_features()

        model = lgb.LGBMClassifier(**params)
        model.fit(X, y)
        logger.info("LightGBM model training completed.")

        model_path = os.path.join(get_root_directory(), "models", "lgbm_Youtube_sentiment.pkl")
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved locally at: {model_path}")

    except Exception as e:
        logger.exception(f"Model building failed: {e}")
        raise

if __name__ == "__main__":
    main()


