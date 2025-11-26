# src/model/model_registration.py

import os
import json
import mlflow
from mlflow.tracking import MlflowClient
import logging

logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('model_registration_errors.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

def get_root_directory():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

def main():
    try:
        info_file = os.path.join(get_root_directory(), "experiment_info.json")
        with open(info_file, "r") as f:
            info = json.load(f)

        run_id = info["run_id"]
        model_name = info["registered_model_name"]
        model_uri = f"runs:/{run_id}/lgbm_model"

        client = MlflowClient()
        mv = mlflow.register_model(model_uri=model_uri, name=model_name)

        client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Staging",
            archive_existing_versions=True
        )

        logger.info(f"Registered & Staged {model_name} version {mv.version}")

    except Exception as e:
        logger.exception("Registration failed")
        raise

if __name__ == "__main__":
    main()





    
