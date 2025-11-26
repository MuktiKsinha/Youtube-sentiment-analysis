# src/model/model_building.py

import os
import pickle
import logging
import yaml
import lightgbm as lgb
import numpy as np
import json
import mlflow
import mlflow.lightgbm

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
# MLflow server
# ---------------------------
MLFLOW_URI = "http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

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
    try:
        with open(os.path.join(root, "params.yaml"), "r") as f:
            params = yaml.safe_load(f)
        return params["model_building"]
    except Exception as e:
        logger.error(f'Failed to load params.yaml: {e}')
        raise

# ---------------------------
# Load BOW features
# ---------------------------
def load_features():
    try:
        with open(path_processed("X_train_BOW.pkl"), "rb") as f:
            X = pickle.load(f)
        with open(path_processed("y_train.pkl"), "rb") as f:
            y = pickle.load(f)
        X = X.astype('float32')
        logger.debug(f"Loaded BOW features: X={X.shape}, y={y.shape}, dtype={X.dtype}")
        return X, y
    except Exception as e:
        logger.error(f"Feature files missing or cannot be loaded: {e}")
        raise

# ---------------------------
# Train model
# ---------------------------
def train_lgbm(X, y, params):
    lgbm_params = params.get("lgbm_params", {})
    model = lgb.LGBMClassifier(**lgbm_params)
    model.fit(X, y)
    logger.debug("LGBM training complete.")
    return model

# ---------------------------
# Save model locally (optional)
# ---------------------------
def save_model(model):
    model_path = os.path.join(get_root_directory(), "models", "lgbm_Youtube_sentiment.pkl")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.debug(f"Model saved locally: {model_path}")
    return model_path

# ---------------------------
# Save model info (MLflow compatible)
# ---------------------------
def save_model_info(run_id: str, artifact_path: str, file_path: str) -> str:
    """
    Save MLflow model info JSON.
    """
    model_info = {
        "run_id": run_id,
        "model_path": artifact_path  # relative path inside run
    }
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(model_info, f, indent=4)
    logger.debug(f"Model info saved to {file_path}")
    return file_path

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        params = load_params()
        X, y = load_features()
        model = train_lgbm(X, y, params)

        # Save model locally (optional)
        save_model(model)

        # Log model to MLflow server and save info JSON
        with mlflow.start_run() as run:
            run_id = run.info.run_id
            artifact_path = "lgbm_Youtube_sentiment.pkl"  # relative path inside run

            # Log model to MLflow server
            mlflow.lightgbm.log_model(model, artifact_path)

            # Save JSON for registration
            info_file = os.path.join(get_root_directory(), "models", "model_info.json")
            save_model_info(run_id, artifact_path, info_file)
            logger.debug(f"Model info JSON saved at {info_file}")

        logger.info("Model building and MLflow logging completed successfully")

    except Exception as e:
        logger.error(f"Model building failed: {e}")
        raise

if __name__ == "__main__":
    main()

