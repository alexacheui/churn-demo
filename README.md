# Churn Demo

A small class project that predicts whether a customer is likely to leave. It is here to show the same basic code in a few different environments: local Python, Docker, Kubernetes, and Databricks.

The sample uses three inputs: tenure in months, monthly spend, and support tickets.

## Files

- `src/train.py` trains a small logistic-regression model.
- `src/app.py` starts a FastAPI prediction service.
- `data/churn_sample.csv` is the tiny sample dataset.
- `Dockerfile` builds the service into an image.
- `k8s/deployment.yaml` runs the service in Kubernetes.
- `databricks/train_churn_notebook.py` is an optional Databricks version of the training step.

## Run locally

Python 3.10 or later is recommended.

```bash
cd churn-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/app.py
```

In another terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tenure_months": 2, "monthly_spend": 120, "support_tickets": 5}'
```

Try different values and compare the risk result.

## Docker

```bash
docker build -t churn-api:demo .
docker run --rm -p 8000:8000 churn-api:demo
```

Use the same `curl` command above once the container starts.

## Kubernetes

With a local cluster running:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl port-forward service/churn-api 8000:80
```

To see Kubernetes replace a stopped pod:

```bash
kubectl delete pod -l app=churn-api
kubectl get pods -w
```

Clean up when finished:

```bash
kubectl delete -f k8s/deployment.yaml
```

## Databricks (optional)

Import `databricks/train_churn_notebook.py` into a workspace, attach a cluster, and run it. The notebook logs a simple MLflow run and model artifact.

## Things to try

1. Add an `age` column and use it in both training and prediction.
2. Reject negative values in the API.
3. Change the Kubernetes replica count from 2 to 3.
4. Make one change on a Git branch and open a pull request.
