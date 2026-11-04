# Churn Demo

A small class project that predicts whether a customer is likely to leave. It is here to show the same basic code in a few different environments: local Python, Docker, Kubernetes, and Databricks.

The sample uses three inputs: tenure in months, monthly spend, and support tickets.

## Files

- `src/train.py` trains a small logistic-regression model.
- `src/app.py` starts a FastAPI prediction service.
- `data/churn_sample.csv` is the tiny sample dataset.
- `requirements.txt` lists the Python packages.
- `Dockerfile` builds the service into an image.
- `k8s/deployment.yaml` runs the service in Kubernetes.
- `databricks/train_churn_notebook.py` is an optional Databricks version of the training step that records three MLflow runs.

## Run locally

Python 3.10 or later is recommended.

Mac / Linux:

```bash
cd churn-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/app.py
```

Windows (PowerShell): use `py -m venv .venv`, then `.venv\Scripts\Activate.ps1`, then the same `pip install`, `python src/train.py`, and `python src/app.py` commands.

`python src/train.py` prints the training accuracy (about 94%). `python src/app.py` keeps running and prints that the server started on port 8000. You may see a harmless warning from scikit-learn or SciPy while training.

Leave that terminal open. The easiest way to try the API is in your browser at <http://127.0.0.1:8000/docs>. Open `POST /predict`, click **Try it out**, and press **Execute**.

Or, in another terminal (Mac / Linux):

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tenure_months": 2, "monthly_spend": 120, "support_tickets": 5}'
```

Expected result: `{"churn_probability":0.816,"risk":"high","model_version":"demo-1"}`

Try different values and compare the risk result. A long-tenure customer with no tickets should come out low.

## Docker

```bash
docker build -t churn-api:demo .
docker run --rm -p 8000:8000 churn-api:demo
```

Use the same `/docs` page or `curl` command above once the container starts.

## Kubernetes

With a local cluster running (Docker Desktop, kind, or minikube). On kind or minikube, load the image into the cluster first:

```bash
kind load docker-image churn-api:demo      # kind
minikube image load churn-api:demo         # minikube
```

Then:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl port-forward service/churn-api 8000:80
```

To see Kubernetes replace a stopped pod, copy one pod name from `kubectl get pods` and delete only that pod:

```bash
kubectl delete pod <pod-name>
kubectl get pods -w
```

Clean up when finished:

```bash
kubectl delete -f k8s/deployment.yaml
```

## Databricks (optional)

Import `databricks/train_churn_notebook.py` into a workspace, attach a cluster, and run it. The notebook trains the model three times with different settings and logs each one as an MLflow run, so you can compare them in the Experiments page.

## Things to try

1. Add an `age` column and use it in both training and prediction.
2. Change the Kubernetes replica count from 2 to 3.
3. Make one change on a Git branch and open a pull request.
4. Tag the version you release, for example `git tag v1`.
