import pytest
import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.tracking import MlflowClient
import os

# Set your remote MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage, holdout_data_path, vectorizer_path", [
    ("Youtube_chrome_plugin_model_final", "staging",
     "data/processed/holdout_data.csv", "bow_vectorizer.pkl")
])
def test_model_performance(model_name, stage, holdout_data_path, vectorizer_path):
    try:
        client = MlflowClient()
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        latest_version = latest_version_info[0].version if latest_version_info else None

        assert latest_version is not None, f"No model found in '{stage}' stage for '{model_name}'"

        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Try loading vectorizer from MLflow first
        vectorizer = None
        try:
            vectorizer_uri = f"{model_uri}/artifacts/bow_vectorizer.pkl"
            vectorizer = mlflow.pyfunc.load_model(vectorizer_uri)
        except Exception:
            print("Vectorizer not found in MLflow → Fallback to local load")

        # Load fallback local vectorizer if needed
        if vectorizer is None:
            assert os.path.exists(vectorizer_path), f"Local vectorizer missing: {vectorizer_path}"
            with open(vectorizer_path, 'rb') as file:
                vectorizer = pickle.load(file)

        # Load Holdout dataset
        holdout_data = pd.read_csv(holdout_data_path)
        X_holdout_raw = holdout_data.iloc[:, 0].fillna("")
        y_holdout = holdout_data.iloc[:, -1]

        # Transform + dtype fix for LightGBM
        X_holdout_vec = vectorizer.transform(X_holdout_raw).astype("float32")

        # Predict
        y_pred = model.predict(X_holdout_vec)

        # Performance metrics
        accuracy = accuracy_score(y_holdout, y_pred)
        precision = precision_score(y_holdout, y_pred, average='weighted', zero_division=1)
        recall = recall_score(y_holdout, y_pred, average='weighted', zero_division=1)
        f1 = f1_score(y_holdout, y_pred, average='weighted', zero_division=1)

        # Threshold requirements
        assert accuracy >= 0.40, f"Accuracy below threshold: {accuracy}"
        assert precision >= 0.40, f"Precision below threshold: {precision}"
        assert recall >= 0.40, f"Recall below threshold: {recall}"
        assert f1 >= 0.40, f"F1 score below threshold: {f1}"

        print(f"Performance test passed for '{model_name}' v{latest_version}")

    except Exception as e:
        pytest.fail(f"Performance test failed: {e}")
