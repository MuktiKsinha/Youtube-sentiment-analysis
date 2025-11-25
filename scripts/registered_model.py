import mlflow

mlflow.set_tracking_uri("http://ec2-13-60-208-36.eu-north-1.compute.amazonaws.com:5000")

from mlflow.tracking import MlflowClient
client = MlflowClient()
print([rm.name for rm in client.search_registered_models()])
