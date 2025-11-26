# scripts/test_model_signature.py

import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("Youtube_chrome_plugin_model_final", "Staging"),
])
def test_model_with_vectorizer(model_name, stage):
    client = MlflowClient()

    # Get latest model version in the stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    assert latest_version_info, f"No model found in stage '{stage}' for '{model_name}'"
    latest_version = latest_version_info[0].version

    # Load the model from MLflow
    model_uri = f"models:/{model_name}/{latest_version}"
    model = mlflow.sklearn.load_model(model_uri)

    # Get the run_id of this model version
    mv = client.get_model_version(model_name, latest_version)
    run_id = mv.run_id

    # Download vectorizer artifact from the correct run
    vectorizer_path = download_artifacts(
        artifact_path="lgbm_model/vectorizer/bow_vectorizer.pkl",
        run_id=run_id
    )
    with open(vectorizer_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Dummy input for testing
    input_text = ["hi how are you"]
    X_input = vectorizer.transform(input_text).astype('float32')

    # Predict using the loaded model
    prediction = model.predict(X_input)

    # Assertions to ensure everything is consistent
    assert X_input.shape[1] == len(vectorizer.get_feature_names_out()), "Feature dimension mismatch"
    assert len(prediction) == 1, "Prediction output length mismatch"

    print(f"Model test passed for {model_name} version {latest_version}")

if __name__ == "__main__":
    # Run the test directly (optional)
    test_model_with_vectorizer("Youtube_chrome_plugin_model_final", "Staging")




