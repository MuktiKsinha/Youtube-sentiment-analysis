import mlflow

from mlflow.tracking import MlflowClient

# set Mlflow tracking uri
mlflow.set_tracking_uri("http://ec2-13-60-52-178.eu-north-1.compute.amazonaws.com:5000/")

# load model and registry

def load_model_from_registry(model_name , model_version):
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
  
  
# example usage
model = load_model_from_registry("Youtube_chrome_plugin_model_final", "1")
print("model loaded successfully")






