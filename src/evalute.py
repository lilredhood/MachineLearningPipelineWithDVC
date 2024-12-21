import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

# Pour le tracking
os.environ["MLFLOW_TRACKING_URI"] = "https://dagshub.com/lilredhood/MachineLearningPipeline.mlflow" # On obtient ce lien via le répo Dagshub
os.environ['MLFLOW_TRACKING_USERNAME'] = "lilredhood"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "8855c7908f32f0a89400ce237c6ea86dc338dd48" # On le trouve dans DVC -> setup credentials dans le répo Gagshub

# Load the parameters from the yaml file

params = yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path) :
    data = pd.read_csv(data_path)
    X = data.drop(columns = ["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/lilredhood/MachineLearningPipeline.mlflow")

    # Load the model from the disk

    model = pickle.load(open(model_path, 'rb'))

    predictions = model.predict(X)
    accuracy = accuracy_score(y, predictions)

    # Log metrics to Mlflow

    mlflow.log_metric("accuracy", accuracy)
    print(f"Model accuracy {accuracy}")

if __name__ == "__main__" :
    evaluate(params["data"], params["model"]) 