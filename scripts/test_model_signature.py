from mlflow.artifacts import download_artifacts
import mlflow
import pickle
import pytest
import pandas as pd
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("Youtube_chrome_plugin_model_final", "Staging"),
])
def test_model_with_vectorizer(model_name, stage):
    client = MlflowClient()
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None
    assert latest_version is not None

    # Load model
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Download vectorizer from the correct path
    vectorizer_path = download_artifacts(
        artifact_path="lgbm_model/vectorizer/bow_vectorizer.pkl",
        artifact_uri=model_uri
    )

    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Dummy input
    input_text = ["hi how are you"]
    X_input = vectorizer.transform(input_text)

    prediction = model.predict(X_input)

    assert X_input.shape[1] == len(vectorizer.get_feature_names_out())
    assert len(prediction) == 1

    print(f"Test passed for {model_name} version {latest_version}")



