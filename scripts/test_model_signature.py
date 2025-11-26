import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage", [
    ("Youtube_chrome_plugin_model_final", "Staging"),
])
def test_model_with_vectorizer(model_name, stage):
    client = MlflowClient()
    
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None
    assert latest_version is not None, f"No model found in stage '{stage}' for '{model_name}'"

    try:
        # Load model
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.sklearn.load_model(model_uri)

        # Download vectorizer using run_id
        mv = client.get_model_version(model_name, latest_version)
        run_id = mv.run_id
        vectorizer_path = download_artifacts("vectorizer/bow_vectorizer.pkl", run_id=run_id)

        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Dummy input
        input_text = ["hi how are you"]
        input_data = vectorizer.transform(input_text)

        # Predict
        prediction = model.predict(input_data)

        # Validation
        assert input_data.shape[1] == len(vectorizer.get_feature_names_out())
        assert len(prediction) == 1

        print(f"Model test passed for {model_name} version {latest_version}")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {str(e)}")


