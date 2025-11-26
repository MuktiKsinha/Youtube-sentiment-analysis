import mlflow
import pytest
import pandas as pd
import pickle
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

# Set your remote tracking URI
mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")


@pytest.mark.parametrize("model_name, stage", [
    ("Youtube_chrome_plugin_model_final", "staging"),
])
def test_model_with_vectorizer(model_name, stage):
    client = MlflowClient()

    # Get latest model version in stage
    latest_version_info = client.get_latest_versions(model_name, stages=[stage])
    latest_version = latest_version_info[0].version if latest_version_info else None

    assert latest_version is not None, f"No model found in stage '{stage}' for '{model_name}'"

    try:
        # Load model from MLflow
        model_uri = f"models:/{model_name}/{latest_version}"
        model = mlflow.pyfunc.load_model(model_uri)

        # Download vectorizer from artifacts
        vectorizer_path = download_artifacts(f"{model_uri}/artifacts/vectorizer/bow_vectorizer.pkl")
        with open(vectorizer_path, "rb") as f:
            vectorizer = pickle.load(f)

        # Dummy test input
        input_text = "hi how are you"
        input_data = vectorizer.transform([input_text])
        input_df = pd.DataFrame(input_data.toarray(), columns=vectorizer.get_feature_names_out())

        # Predict
        prediction = model.predict(input_df)

        # Validation
        assert input_df.shape[1] == len(vectorizer.get_feature_names_out())
        assert len(prediction) == 1  

        print(f"Model test passed for {model_name} version {latest_version}")

    except Exception as e:
        pytest.fail(f"Model test failed with error: {str(e)}")

