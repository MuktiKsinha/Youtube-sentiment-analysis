import mlflow

mlflow.set_tracking_uri("http://ec2-16-171-11-131.eu-north-1.compute.amazonaws.com:5000")

from mlflow.tracking import MlflowClient
client = MlflowClient()
print([rm.name for rm in client.search_registered_models()])
