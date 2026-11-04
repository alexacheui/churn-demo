"""Train and save a tiny churn classifier for the classroom demo."""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["tenure_months", "monthly_spend", "support_tickets"]


def train() -> float:
    data = pd.read_csv(ROOT / "data" / "churn_sample.csv")
    # Scaling the inputs and a moderate C keep probabilities between 0 and 1
    # instead of jumping straight to 0% or 100% on this tiny data set.
    model = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, random_state=42, max_iter=1_000))
    model.fit(data[FEATURES], data["churned"])

    accuracy = accuracy_score(data["churned"], model.predict(data[FEATURES]))
    output_dir = ROOT / "artifacts"
    output_dir.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES, "version": "demo-1"}, output_dir / "model.joblib")
    print(f"Saved artifacts/model.joblib (training accuracy: {accuracy:.0%})")
    return accuracy


if __name__ == "__main__":
    train()

