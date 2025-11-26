# src/model/model_evaluation.py

import os
import pickle
import pandas as pd
import json
import logging
import mlflow
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from mlflow.models import infer_signature
import yaml

try:
    import seaborn as sns
except:
    sns = None

# ---------------------------
# Logging
# ---------------------------
logger = logging.getLogger('model_evaluation')
logger.setLevel(logging.DEBUG)
if not logger.hasHandlers():
    ch = logging.StreamHandler()
    fh = logging.FileHandler('model_evaluation_errors.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(ch)
    logger.addHandler(fh)

# ---------------------------
# Helpers
# ---------------------------
def get_root_directory():
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))

def path_processed(filename: str):
    return os.path.join(get_root_directory(), 'data/processed', filename)

def load_model(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_vectorizer(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_data(path):
    df = pd.read_csv(path)
    df.fillna('', inplace=True)
    return df

# ---------------------------
# Main
# ---------------------------
def main():
    try:
        root = get_root_directory()
        mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")
        mlflow.set_experiment("Youtube_sentiment_evaluation_final")

        # Load model, vectorizer, test data
        model_file = os.path.join(root, "models", "lgbm_Youtube_sentiment.pkl")
        vectorizer_file = path_processed("bow_vectorizer.pkl")
        test_file = os.path.join(root, "data/interim/test_processed.csv")

        model = load_model(model_file)
        vectorizer = load_vectorizer(vectorizer_file)
        test_df = load_data(test_file)

        # Load parameters for logging
        with open(os.path.join(root, "params.yaml"), "r") as f:
            params_all = yaml.safe_load(f)
        model_params = params_all.get("model_building", {}).get("lgbm_params", {})

        # Prepare reports directory for DVC outputs
        reports_dir = os.path.join(root, 'reports')
        os.makedirs(reports_dir, exist_ok=True)

        with mlflow.start_run() as run:

            # -------------------------
            # Log model parameters
            # -------------------------
            for k, v in model_params.items():
                mlflow.log_param(k, v)

            # Transform test set
            X_test = vectorizer.transform(test_df["clean_comment"].values).astype('float32')
            y_test = test_df["category"].values

            # Evaluate
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred)

            # -------------------------
            # Log metrics
            # -------------------------
            for label, metrics_dict in report.items():
                if isinstance(metrics_dict, dict):
                    for m_name, m_value in metrics_dict.items():
                        if m_name != "support":
                            mlflow.log_metric(f"{label}_{m_name}", float(m_value))
                elif label == "accuracy":
                    mlflow.log_metric("accuracy", float(metrics_dict))

            # -------------------------
            # Save DVC outputs
            # -------------------------
            classification_report_path = os.path.join(reports_dir, 'classification_report.json')
            with open(classification_report_path, 'w') as f:
                json.dump(report, f, indent=4)

            cm_file_path = os.path.join(reports_dir, 'confusion_matrix_test_set.png')
            plt.figure(figsize=(8, 6))
            if sns:
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            else:
                plt.imshow(cm, cmap='Blues')
            plt.title('Confusion Matrix - Test Set')
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            plt.savefig(cm_file_path)
            plt.close()

            # Save experiment info for registration
            experiment_info_path = os.path.join(root, 'experiment_info.json')
            info = {"run_id": run.info.run_id, "registered_model_name": "Youtube_chrome_plugin_model_final"}
            with open(experiment_info_path, 'w') as f:
                json.dump(info, f, indent=4)

            # -------------------------
            # Log artifacts to MLflow
            # -------------------------
            mlflow.log_artifact(classification_report_path)
            mlflow.log_artifact(cm_file_path)
            mlflow.log_artifact(experiment_info_path)

            # -------------------------
            # Log model and include vectorizer inside model artifact
            # -------------------------
            signature = infer_signature(X_test[:5].toarray(), model.predict(X_test[:5]))
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="lgbm_model",
                signature=signature,
                extra_files={ "bow_vectorizer.pkl": vectorizer_file }
            )

            logger.info("Model evaluation logged successfully.")

    except Exception as e:
        logger.exception(e)
        raise

if __name__ == "__main__":
    main()







