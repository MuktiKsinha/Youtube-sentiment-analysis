import mlflow
from mlflow import pyfunc

MLFLOW_TRACKING_URI = "http://ec2-51-20-65-11.eu-north-1.compute.amazonaws.com:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

def load_model_from_registry(model_name, version):
    model_uri = f"models:/{model_name}/{version}"
    return pyfunc.load_model(model_uri)

model = load_model_from_registry("Youtube_chrome_plugin_model", "3")
print("model loaded successfully")