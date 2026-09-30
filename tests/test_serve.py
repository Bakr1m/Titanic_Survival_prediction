"""Serving contract tests: leak-free schema, lazy loading, validation.

Uses a tiny stand-in pipeline (never the production artifact) so tests
are hermetic: they pass on a clean checkout with no data/ or models/.
"""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import src.serve as serve
from src.serve import app

LEAKY_COLS = {"alive", "class", "who", "embark_town", "alone"}

VALID_PASSENGER = {
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 100.0,
    "embarked": "S",
    "adult_male": False,
    "deck": "C",
    "family_size": 1,
    "is_alone": 1,
    "fare_per_person": 100.0,
    "age_bin": "Young Adult",
}


@pytest.fixture()
def client(monkeypatch):
    """TestClient wired to a tiny stand-in pipeline on synthetic rows."""
    df = pd.DataFrame(
        [
            {**VALID_PASSENGER, "sex": "female", "survived": 1},
            {**VALID_PASSENGER, "sex": "male", "age": 40, "survived": 0},
            {**VALID_PASSENGER, "pclass": 3, "fare": 8.0, "survived": 0},
            {**VALID_PASSENGER, "age": 5, "survived": 1},
        ]
    )
    X, y = df.drop(columns=["survived"]), df["survived"]
    num = X.select_dtypes(include="number").columns.tolist()
    cat = [c for c in X.columns if c not in num]
    pre = ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer()), ("sc", StandardScaler())]), num),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                              ("enc", OneHotEncoder(handle_unknown="ignore"))]), cat),
        ]
    )
    pipe = Pipeline([("preprocessor", pre),
                     ("classifier", RandomForestClassifier(n_estimators=5, random_state=42))])
    pipe.fit(X, y)
    monkeypatch.setattr(serve, "_pipe", pipe)
    return TestClient(app)


def test_health():
    assert TestClient(app).get("/health").json() == {"status": "healthy"}


def test_predict_valid(client):
    r = client.post("/predict", json=VALID_PASSENGER)
    assert r.status_code == 200
    body = r.json()
    assert body["survived"] in (0, 1)
    assert 0.0 <= body["survival_probability"] <= 1.0
    assert body["risk_category"] in ("high", "medium", "low")


def test_leaky_columns_rejected(client):
    """Label-derived fields must fail at the contract, never reach the model."""
    for col in LEAKY_COLS:
        payload = dict(VALID_PASSENGER, **{col: "x"})
        r = client.post("/predict", json=payload)
        assert r.status_code == 422, col


def test_schema_has_no_leaky_fields():
    fields = set(serve.PassengerInput.model_fields)
    assert not (fields & (LEAKY_COLS | {"survived"})), fields & LEAKY_COLS
    assert len(fields) == 13


def test_invalid_age_rejected(client):
    r = client.post("/predict", json={**VALID_PASSENGER, "age": -5})
    assert r.status_code == 422


def test_expected_features_come_from_fitted_transformer(client):
    feats = serve.expected_features(serve.get_pipeline())
    assert set(feats) == set(VALID_PASSENGER)
    assert not (set(feats) & LEAKY_COLS)
