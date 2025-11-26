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
try:
    import seaborn as sns
except Exception:
    sns = None
import json
from mlflow.models import infer_signature
from scipy.sparse import csr_matrix

# ---------------------------
# Logging configuration
# ---------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)

if not logger.hasHandlers():
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('model_evaluation_errors.log')
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

def path_processed(filename: str) -> str:
    return os.path.join(get_root_directory(), 'data/processed', filename)

def model_path(filename: str) -> str:
    return os.path.join(get_root_directory(), 'models', filename)

def load_data(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        df.fillna('', inplace=True)
        logger.debug('Data loaded and NaNs filled from %s', path)
        return df
    except Exception as e:
        logger.error('Error loading dataset: %s', e)
        raise

def load_model(path: str):
    try:
        with open(path, 'rb') as file:
            model = pickle.load(file)
        logger.debug('Model loaded from %s', path)
        return model
    except Exception as e:
        logger.error('Error loading model from %s: %s', path, e)
        raise

def load_vectorizer(path: str) -> CountVectorizer:
    try:
        with open(path, 'rb') as file:
            vectorizer = pickle.load(file)
        logger.debug('Vectorizer loaded from %s', path)
        return vectorizer
    except Exception as e:
        logger.error('Error loading vectorizer from %s: %s', path, e)
        raise

def load_params(root: str) -> dict:
    try:
        with open(os.path.join(root, "params.yaml"), "r") as f:
            params = yaml.safe_load(f)
        model_params = params.get("model_building", {})
        return model_params.get("lgbm_params", {})
    except Exception as e:
        logger.error("Failed to load params.yaml: %s", e)
        raise

def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        return report, cm
    except Exception as e:
        logger.error('Error during model evaluation: %s', e)
        raise

def log_confusion_matrix(cm, dataset_name):
    plt.figure(figsize=(8, 6))
    if sns:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    else:
        plt.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                color = "white" if cm[i, j] > cm.max()/2 else "black"
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', color=color)

    plt.title(f'Confusion Matrix - {dataset_name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    reports_dir = os.path.join(get_root_directory(), 'reports')
    os.makedirs(reports_dir, exist_ok=True)
    cm_file_path = os.path.join(reports_dir, f'confusion_matrix_{dataset_name}.png')
    plt.savefig(cm_file_path)
    mlflow.log_artifact(cm_file_path)
    plt.close()

def log_all_params(params: dict):
    def _log_dict(d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                _log_dict(v, prefix=f"{prefix}{k}_")
            else:
                mlflow.log_param(f"{prefix}{k}", v)
    _log_dict(params)

def log_metrics(report: dict):
    for label, metrics_dict in report.items():
        if isinstance(metrics_dict, dict):
            for m_name, m_value in metrics_dict.items():
                if m_name != "support":
                    mlflow.log_metric(f"test_{label}_{m_name}", float(m_value))
        elif label == "accuracy":
            mlflow.log_metric("test_accuracy", float(metrics_dict))

def save_model_info(run_id: str, registered_model_name: str, model_version: int, file_path: str) -> None:
    try:
        model_info = {
            'run_id': run_id,
            'registered_model_name': registered_model_name,
            'model_version': model_version
        }
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logger.debug('Model info saved to %s', file_path)
    except Exception as e:
        logger.error('Error occurred while saving the model info: %s', e)
        raise

# ---------------------------
# Main evaluation
# ---------------------------
def main():
    try:
        root = get_root_directory()
        models_dir = os.path.join(root, "models")
        processed_dir = os.path.join(root, "data", "processed")
        interim_dir = os.path.join(root, "data", "interim")

        # Load params
        params = load_params(root)

        # MLflow setup
        mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")
        mlflow.set_experiment("Youtube_sentiment_evaluation_final")

        with mlflow.start_run() as run:
            log_all_params(params)

            # Load model, vectorizer, test data
            model_file = os.path.join(models_dir, 'lgbm_Youtube_sentiment.pkl')
            vectorizer_file = os.path.join(processed_dir, 'bow_vectorizer.pkl')
            X_train_file = path_processed("X_train_BOW.pkl")
            y_train_file = path_processed("y_train.pkl")
            test_file = os.path.join(interim_dir, 'test_processed.csv')

            model = load_model(model_file)
            vectorizer = load_vectorizer(vectorizer_file)
            test_df = load_data(test_file)

            if 'clean_comment' not in test_df or 'category' not in test_df:
                raise KeyError("test_processed.csv must contain 'clean_comment' and 'category' columns")

            # Transform test comments using BOW vectorizer
            X_test = vectorizer.transform(test_df["clean_comment"].values).astype('float32')
            y_test = test_df["category"].values

            # Evaluate model
            report, cm = evaluate_model(model, X_test, y_test)

            # Save and log classification report
            reports_dir = os.path.join(root, 'reports')
            os.makedirs(reports_dir, exist_ok=True)
            classification_report_path = os.path.join(reports_dir, 'classification_report.json')
            with open(classification_report_path, 'w') as f:
                json.dump(report, f, indent=4)
            mlflow.log_artifact(classification_report_path)

            # Log metrics & confusion matrix
            log_metrics(report)
            log_confusion_matrix(cm, "test_set")

            # -----------------------------------
            # ✔ Infer signature for MLflow Model
            # -----------------------------------
            sample_size = min(10, len(test_df))
            sample_text = test_df["clean_comment"].iloc[:sample_size].tolist()

            sample_features = vectorizer.transform(sample_text).astype('float32')

            signature = infer_signature(
                sample_features.toarray(),
                model.predict(sample_features)
            )

            # -----------------------------
            # Log model & register in MLflow
            # -----------------------------
            registered_model_name = "Youtube_chrome_plugin_model_final"

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",
                signature=signature,
                registered_model_name=registered_model_name
            )

            # Get registered model version
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(registered_model_name)
            model_version = max([int(v.version) for v in latest_versions]) if latest_versions else 1

            # Save model info
            save_model_info(run.info.run_id, registered_model_name, model_version, 'experiment_info.json')

            # Log vectorizer as artifact
            mlflow.log_artifact(vectorizer_file, artifact_path="vectorizer")

            # Log training features as artifacts
            mlflow.log_artifact(X_train_file, artifact_path="features")
            mlflow.log_artifact(y_train_file, artifact_path="features")

            mlflow.set_tag("model_type", "LightGBM")
            mlflow.set_tag("task", "Sentiment Analysis")

            logger.info("Model evaluation completed successfully. Run ID: %s", run.info.run_id)

    except Exception as e:
        logger.exception("Evaluation failed: %s", e)
        raise

if __name__ == '__main__':
    main()


