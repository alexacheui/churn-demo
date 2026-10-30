# Databricks notebook source
# COMMAND ----------
# Upload data/churn_sample.csv to a Unity Catalog volume or replace this with
# your own table. This example uses a short in-memory data set to stay portable.
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.DataFrame(
    [
        [1, 130, 6, 1], [2, 110, 5, 1], [8, 75, 3, 1], [12, 65, 2, 0],
        [24, 90, 1, 0], [36, 115, 1, 0], [48, 125, 0, 0], [4, 60, 6, 1],
    ],
    columns=["tenure_months", "monthly_spend", "support_tickets", "churned"],
)
features = ["tenure_months", "monthly_spend", "support_tickets"]

with mlflow.start_run():
    model = LogisticRegression(random_state=42, max_iter=1_000)
    model.fit(data[features], data["churned"])
    accuracy = model.score(data[features], data["churned"])
    mlflow.log_param("algorithm", "logistic_regression")
    mlflow.log_metric("training_accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")
    print(f"Training accuracy: {accuracy:.0%}")

# COMMAND ----------
# Discussion prompt: once a model is approved in MLflow Model Registry,
# a CI/CD pipeline can build a Docker image containing that model version
# and update the Kubernetes deployment.
