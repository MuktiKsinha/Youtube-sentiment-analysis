import pytest
import pandas as pd
import mlflow
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# Set MLflow remote tracking URI
mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage, holdout_data_path", [
    ("Youtube_chrome_plugin_model_final", "staging", "data/processed/holdout_data.csv"),
])
def test_model_performance(model_name, stage, holdout_data_path):
    try:
        client = MlflowClient()

        # Get latest version in stage
        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        assert latest_version_info, f"No model found in '{stage}' for '{model_name}'"
        version = latest_version_info[0].version

        # Load model from registry
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Fetch run where the model + vectorizer were logged
        mv = client.get_model_version(model_name, version)
        run_id = mv.run_id

        # Download vectorizer from artifacts
        vectorizer_path = download_artifacts(
            artifact_path="lgbm_model/vectorizer/bow_vectorizer.pkl",
            run_id=run_id
        )
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Load evaluation dataset
        df = pd.read_csv(holdout_data_path)
        X_raw = df.iloc[:, 0].fillna("")  # text column
        y_true = df.iloc[:, -1]  # label column

        # Vectorize holdout data
        X_vec = vectorizer.transform(X_raw).astype("float32")

        # Predict from model
        y_pred = model.predict(X_vec)

        # Evaluate performance
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

        print("\nPerformance Results:",
              f"\nAccuracy: {accuracy:.3f}",
              f"\nPrecision: {precision:.3f}",
              f"\nRecall: {recall:.3f}",
              f"\nF1 Score: {f1:.3f}")

        # Minimum performance thresholds for promotion
        assert accuracy >= 0.40
        assert precision >= 0.40
        assert recall >= 0.40
        assert f1 >= 0.40

        print(f"\n✔ Performance test PASSED for {model_name} v{version}")

    except Exception as e:
        pytest.fail(f"Model performance test failed: {e}")

