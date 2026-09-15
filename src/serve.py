"""
FastAPI serving script for Titanic Survival model
"""
import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

app = FastAPI(title="Titanic Survival Prediction API")

# Load pipeline at startup
pipeline = joblib.load("models/titanic_pipeline.joblib")

# Feature columns (must match training)
FEATURE_COLS = [
    'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
    'pclass_str', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone',
    'family_size', 'is_alone', 'fare_per_person', 'age_bin'
]


class PassengerInput(BaseModel):
    """Input schema for prediction"""
    pclass: int = Field(..., ge=1, le=3, description="Passenger class (1=1st, 2=2nd, 3=3rd)")
    sex: str = Field(..., description="Sex: male or female")
    age: float = Field(..., ge=0, le=100, description="Age in years")
    sibsp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    fare: float = Field(..., ge=0, description="Passenger fare")
    embarked: str = Field(..., description="Port of embarkation: S, C, or Q")
    pclass_str: str = Field(..., description="Class: First, Second, Third", alias="class")
    who: str = Field(..., description="Who: man, woman, child")
    adult_male: bool = Field(..., description="Is adult male")
    deck: str = Field(..., description="Deck: A, B, C, D, E, F, G, Unknown")
    embark_town: str = Field(..., description="Embarkation town")
    alive: str = Field(..., description="Alive: yes or no")
    alone: bool = Field(..., description="Traveling alone")
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


@app.post("/predict")
def predict(passenger: PassengerInput):
    try:
        # Convert to DataFrame with correct column order
        input_dict = passenger.model_dump(by_alias=True)
        input_df = pd.DataFrame([input_dict])
        input_df.columns = [
            'pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked',
            'class', 'who', 'adult_male', 'deck', 'embark_town', 'alive', 'alone',
            'family_size', 'is_alone', 'fare_per_person', 'age_bin'
        ]
        
        # Predict
        proba = pipeline.predict_proba(input_df)[0, 1]
        pred = int(proba >= 0.5)
        
        return {
            "survived": pred,
            "survival_probability": float(proba),
            "death_probability": float(1 - proba),
            "risk_category": "high" if proba >= 0.7 else "medium" if proba >= 0.4 else "low"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)