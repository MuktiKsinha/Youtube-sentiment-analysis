import pytest
import pandas as pd
import mlflow
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

@pytest.mark.parametrize("model_name, stage, holdout_data_path", [
    ("Youtube_chrome_plugin_model_final", "staging", "data/processed/holdout_data.csv"),
])
def test_model_performance(model_name, stage, holdout_data_path):

    try:
        client = mlflow.tracking.MlflowClient()

        latest_version_info = client.get_latest_versions(model_name, stages=[stage])
        assert latest_version_info, f"No model in stage '{stage}' for '{model_name}'"
        
        version = latest_version_info[0].version
        model_uri = f"models:/{model_name}/{version}"

        # Load model
        model = mlflow.pyfunc.load_model(model_uri)

        # Load vectorizer artifact logged with the model
        vectorizer_uri = f"{model_uri}/artifacts/bow_vectorizer.pkl"
        vectorizer = mlflow.pyfunc.load_model(vectorizer_uri)

        # Load holdout test data
        df = pd.read_csv(holdout_data_path)
        X_raw = df.iloc[:, 0].fillna("")  # text column
        y_true = df.iloc[:, -1]  # labels

        # Transform input text using loaded vectorizer
        X_vec = vectorizer.transform(X_raw)
        X_df = pd.DataFrame(X_vec.toarray(), columns=vectorizer.get_feature_names_out())

        # Predict
        y_pred = model.predict(X_df)

        # Performance metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average="weighted", zero_division=1)
        recall = recall_score(y_true, y_pred, average="weighted", zero_division=1)
        f1 = f1_score(y_true, y_pred, average="weighted", zero_division=1)

        # Expected thresholds
        assert accuracy >= 0.40
        assert precision >= 0.40
        assert recall >= 0.40
        assert f1 >= 0.40

        print(f"Performance test passed for {model_name} v{version}")

    except Exception as e:
        pytest.fail(f"Model performance test failed: {e}")

