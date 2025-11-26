# src/model/model_evaluation.py

import sys
import os
import numpy as np
import pandas as pd
import pickle
import logging
import yaml
import mlflow
from mlflow.sklearn import log_model
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
import json

try:
    import seaborn as sns
except Exception:
    sns = None

logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    file_handler = logging.FileHandler('model_evaluation_errors.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def get_root_directory() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    return df

def load_model(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_vectorizer(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)

def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    if sns:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        plt.imshow(cm, cmap='Blues')

    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    cm_file = os.path.join(get_root_directory(), 'reports', f'conf_matrix_{dataset_name}.png')
    os.makedirs(os.path.dirname(cm_file), exist_ok=True)
    plt.savefig(cm_file)
    mlflow.log_artifact(cm_file)
    plt.close()

def main():
    try:
        root = get_root_directory()
        mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")
        mlflow.set_experiment("Youtube_sentiment_evaluation_final")

        # Load artifacts
        model = load_model(os.path.join(root, "models", "lgbm_Youtube_sentiment.pkl"))
        vectorizer = load_vectorizer(path_processed("bow_vectorizer.pkl"))
        test_df = load_data(os.path.join(root, "data/interim/test_processed.csv"))

        with mlflow.start_run() as run:
            X_test = vectorizer.transform(test_df["clean_comment"].values).astype('float32')
            y_test = test_df["category"].values

            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # Log metrics
            mlflow.log_metric("test_accuracy", report.get("accuracy", 0))

            log_confusion_matrix(cm, "test")

            signature = infer_signature(
                X_test[:5].toarray(),
                model.predict(X_test[:5])
            )

            registered_model_name = "Youtube_chrome_plugin_model_final"

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",
                signature=signature,
                registered_model_name=registered_model_name
            )

            # Save info for registration script
            info = {
                "run_id": run.info.run_id,
                "registered_model_name": registered_model_name
            }
            with open("experiment_info.json", "w") as f:
                json.dump(info, f, indent=4)

            mlflow.log_artifact("experiment_info.json")

            logger.info("Model evaluation logged successfully.")

    except Exception as e:
        logger.exception(e)
        raise

if __name__ == "__main__":
    main()




