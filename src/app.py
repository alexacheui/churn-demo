"""A small prediction API that loads the trained churn model."""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
bundle_path = ROOT / "artifacts" / "model.joblib"
if not bundle_path.exists():
    raise RuntimeError("Model is missing. Run: python src/train.py")
bundle = joblib.load(bundle_path)

app = FastAPI(title="Churn Prediction API", version=bundle["version"])


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=120)
    monthly_spend: float = Field(ge=0, le=10_000)
    support_tickets: int = Field(ge=0, le=100)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": bundle["version"]}


@app.post("/predict")
def predict(customer: Customer):
    values = pd.DataFrame([customer.model_dump()])[bundle["features"]]
    probability = float(bundle["model"].predict_proba(values)[0][1])
    return {
        "churn_probability": round(probability, 3),
        "risk": "high" if probability >= 0.5 else "low",
        "model_version": bundle["version"],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
