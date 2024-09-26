import os
import warnings
import sys

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import ElasticNet
from urllib.parse import urlparse

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def eval_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(42)

    csv_url = ("https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv")

    try:
        data = pd.read_csv(csv_url, sep=";")

    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )


    train, test = train_test_split(data)

    X_train = train.drop("quality", axis=1)
    X_test = test.drop("quality", axis=1)
    y_train = train["quality"]
    y_test = test["quality"]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run():
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(X_train, y_train)

        predicted_qualities = model.predict(X_test)

        (rmse, mae, r2) = eval_metrics(y_test, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f})".format(alpha, l1_ratio))
        print(" RMSE: %s" %rmse)
        print(" MAE: %s" %mae)
        print(" R2: %s" %r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("rmse", rmse)
        mlflow.log_param("r2", r2)
        mlflow.log_metric("rmse", rmse)

        predictions = model.predict(X_train)
        signature = infer_signature(X_train, predictions)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_url_type_store != "file":

            mlflow.sklearn.log_model(
                model, "model",
                registered_model= "ElasticnetWineModel",
                signature=signature)
        else:
            mlflow.sklearn.log_model(model, "model", signature=signature)


