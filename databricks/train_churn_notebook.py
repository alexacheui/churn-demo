# Databricks notebook source
# This notebook repeats the training step from src/train.py, but every run is
# recorded in MLflow so a team can compare runs later.
# The data below is the same as data/churn_sample.csv, pasted in so the notebook
# runs without uploading a file. Replace it with your own table if you like.

# COMMAND ----------
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

features = ["tenure_months", "monthly_spend", "support_tickets"]
data = pd.DataFrame(
    [
        [1, 130, 6, 1],
        [2, 110, 5, 1],
        [3, 95, 4, 1],
        [5, 85, 4, 1],
        [8, 75, 3, 1],
        [10, 70, 2, 0],
        [12, 65, 2, 0],
        [18, 80, 1, 0],
        [24, 90, 1, 0],
        [30, 100, 0, 0],
        [36, 115, 1, 0],
        [48, 125, 0, 0],
        [6, 140, 5, 1],
        [20, 55, 3, 0],
        [4, 60, 6, 1],
        [42, 75, 0, 0],
    ],
    columns=features + ["churned"],
)

# COMMAND ----------
# Train three times with different settings. Each run shows up in MLflow with its
# own parameter and metric, so you can compare them side by side.
for c_value in [0.03, 0.3, 1.0]:
    with mlflow.start_run(run_name=f"logreg_C_{c_value}"):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, random_state=42, max_iter=1_000))
        model.fit(data[features], data["churned"])
        accuracy = model.score(data[features], data["churned"])
        mlflow.log_param("algorithm", "logistic_regression")
        mlflow.log_param("C", c_value)
        mlflow.log_metric("training_accuracy", accuracy)
        mlflow.sklearn.log_model(model, "model")
        print(f"C={c_value}: training accuracy {accuracy:.0%}")

# COMMAND ----------
# Discussion prompt: once a model version is approved in the MLflow Model Registry,
# a CI/CD pipeline can build a Docker image containing that version
# and update the Kubernetes deployment.
# (Optional: pass registered_model_name="churn_model" to log_model to register it.
# Registry naming rules depend on your workspace setup.)
