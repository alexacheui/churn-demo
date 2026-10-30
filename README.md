# Churn MLOps Demo

A small end-to-end project for showing how an ML model moves from code to a running service. The model is intentionally simple: it estimates churn risk from `tenure_months`, `monthly_spend`, and `support_tickets`.

## The MLOps story

```text
Git repository -> train model -> Docker image -> Kubernetes service
                       ^
                       | (optional: run the same training code in Databricks)
```

| Tool | What it does here |
| --- | --- |
| Git | Tracks the code, model recipe, and deployment configuration. |
| Databricks | Can run the training job against larger data and track the experiment. |
| Docker | Packages the prediction API with its exact Python dependencies. |
| Kubernetes | Keeps one or more API containers running and exposes them to clients. |

## 1. Run locally (no Docker required)

Requires Python 3.10+.

```bash
cd churn-mlops-demo
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
python src/app.py
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/health
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"tenure_months": 2, "monthly_spend": 120, "support_tickets": 5}'
```

Change the numbers and ask students which customer they expect to have the larger risk.

## 2. Git mini-demo

Initialize this folder as its own repository (if it is not already in one):

```bash
git init
git add .
git commit -m "Create churn prediction service"
git checkout -b feature/add-model-version
```

Have students add a `model_version` field to the response, commit it, then open a pull request or explain the review that would happen before merging. Do **not** commit `artifacts/model.joblib`; it is generated and ignored. In a real system, the model registry stores the approved artifact.

## 3. Docker demo

Docker runs training while building the image, then starts the API:

```bash
docker build -t churn-api:demo .
docker run --rm -p 8000:8000 churn-api:demo
```

Repeat the `curl` command above. The point to make: the same image runs consistently on every machine that has Docker.

## 4. Kubernetes demo

With Docker Desktop Kubernetes, minikube, or another local cluster enabled:

```bash
kubectl apply -f k8s/deployment.yaml
kubectl get pods
kubectl port-forward service/churn-api 8000:80
```

Then use the same `curl` request. To demonstrate self-healing, delete a pod and watch Kubernetes replace it:

```bash
kubectl delete pod -l app=churn-api
kubectl get pods -w
```

For a local cluster, ensure the image `churn-api:demo` is available to that cluster. Docker Desktop normally shares the local Docker image store; minikube may need `minikube image load churn-api:demo`.

Clean up after class:

```bash
kubectl delete -f k8s/deployment.yaml
```

## 5. Optional Databricks exercise

Import `databricks/train_churn_notebook.py` into a Databricks workspace, attach any cluster, and run it. It uses the same training approach, logs a run to MLflow, and registers an artifact. It is independent of the web API so students without workspace permissions can skip it.

## Suggested 15-minute live demo

1. **2 min:** Show the repository tree and explain that Git holds the recipe, not the large data/model files.
2. **3 min:** Run `src/train.py`, open `data/churn_sample.csv`, and make one local prediction.
3. **3 min:** Build and run the Docker image; repeat the exact request.
4. **4 min:** Apply the Kubernetes manifests, port-forward, and show the two pods.
5. **2 min:** Delete one pod and watch Kubernetes restore the desired state.
6. **1 min:** Open the Databricks notebook and connect its MLflow run to the model-training stage.

## Student challenges

1. Add an `age` feature to the dataset, API request, and training code.
2. Add input validation that rejects a negative spend or ticket count.
3. Change `replicas` from 2 to 3 and observe the new pod.
4. Use a Git branch and pull request to propose one of the changes.

