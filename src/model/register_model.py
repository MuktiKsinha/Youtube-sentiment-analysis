# src/model/model_registration.py

import json
import mlflow
from mlflow.tracking import MlflowClient
import logging
import os

# --------------------------
# Logging
# --------------------------
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------
# MLflow server
# ---------------------------
MLFLOW_URI = "http://ec2-13-60-208-36.eu-north-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_URI)

# ---------------------------
# Helper functions
# ---------------------------
def get_root_directory() -> str:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(current_dir, '../../'))

def load_model_info(file_path: str) -> dict:
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('Model info loaded from %s', file_path)
        return model_info
    except Exception as e:
        logger.error('Error loading model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    try:
        # Correct MLflow URI: run_id + relative artifact path
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.info("Registering model from URI: %s", model_uri)

        #register the model
        
        client = MlflowClient()
        model_version = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(f"Model registered: {model_name}, version {model_version.version}")

        # Transition new version to 'Staging' and archive old Staging versions
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging",
            archive_existing_versions=True
        )
        logger.info(f"Model {model_name} version {model_version.version} transitioned to Staging.")

    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        root = get_root_directory()
        model_info_path = os.path.join(root, "models", "model_info.json")

        model_info = load_model_info(model_info_path)
        model_name = "Youtube_chrome_plugin_model_final"

        register_model(model_name, model_info)

    except Exception as e:
        logger.error('Failed to register the model: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()




    
