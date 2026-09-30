"""
FastAPI serving script for Titanic Survival model.

Serves the 13 leak-free features the pipeline was trained on.
Label-derived columns (alive, class, who, embark_town, alone) are
rejected at the contract level (extra="forbid"): callers must not
send them, and the model never sees them.
"""
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "titanic_pipeline.joblib"

app = FastAPI(title="Titanic Survival Prediction API")

_pipe = None


def get_pipeline():
    """Load once on first request (never at import: hermetic tests)."""
    global _pipe
    if _pipe is None:
        if not MODEL_PATH.exists():
            raise HTTPException(
                status_code=503,
                detail=f"model artifact missing: {MODEL_PATH}",
            )
        _pipe = joblib.load(MODEL_PATH)
    return _pipe


def expected_features(pipeline) -> list:
    """Raw input columns the fitted preprocessor was built on.

    Derived from the fitted ColumnTransformer, never hardcoded —
    a retrain that changes features updates the contract automatically.
    """
    pre = pipeline.named_steps["preprocessor"]
    cols = []
    for _, _, columns in pre.transformers_:
        cols.extend(list(columns))
    return cols


class PassengerInput(BaseModel):
    """Leak-free passenger facts. Label-derived fields (alive, class,
    who, embark_town, alone) are forbidden — see model card."""

    model_config = ConfigDict(extra="forbid")

    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1=1st, 2=2nd, 3=3rd)")
    sex: str = Field(..., description="Sex: male or female")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: str = Field(..., description="Port of embarkation: S, C, or Q")
    adult_male: bool = Field(..., description="Is adult male")
    deck: str = Field(..., description="Deck: A, B, C, D, E, F, G, Unknown")
    family_size: int = Field(..., ge=1, description="Family size including passenger")
    is_alone: int = Field(..., ge=0, le=1, description="Is alone (0/1)")
    fare_per_person: float = Field(..., ge=0, description="Fare per person")
    age_bin: str = Field(..., description="Age bin: Child, Teen, Young Adult, Adult, Senior")


class PredictionResponse(BaseModel):
    survived: int
    survival_probability: float
    death_probability: float
    risk_category: str


def get_risk_category(probability: float) -> str:
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    return "low"


@app.get("/")
def root():
    return {"message": "Titanic Survival Prediction API", "status": "ready"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictionResponse)
def predict(passenger: PassengerInput):
    pipeline = get_pipeline()
    try:
        input_df = pd.DataFrame([passenger.model_dump()])
        input_df = input_df[expected_features(pipeline)]
        proba = pipeline.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)
        return {
            "survived": pred,
            "survival_probability": float(proba),
            "death_probability": float(1 - proba),
            "risk_category": get_risk_category(float(proba)),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
