# Phase 1: Titanic Survival Prediction — Mini Project

**Days 1–10 | Crash Course Foundations**

This is the Phase 1 capstone mini-project from the 60-Day ML Engineer Roadmap. It demonstrates the complete ML lifecycle: data profiling → cleaning → EDA → modeling → evaluation → API → Docker deployment.

## Business Context

Predict survival on the Titanic using passenger demographics and ticket information. While not a healthcare problem, this serves as the "dress rehearsal" for the 5 real healthcare projects that follow.

## Dataset

- **Source**: Seaborn's built-in Titanic dataset (891 passengers)
- **Target**: `survived` (binary: 0=died, 1=survived)
- **Features**: 13 features after cleaning (demographics, ticket info, engineered features) — leaky columns removed (`alive`, `class`, `who`, `embark_town`, `alone`)
- **Class Balance**: 61.6% died, 38.4% survived

## Key Results

| Model | ROC-AUC | PR-AUC | F1 | Accuracy |
|-------|---------|--------|-----|----------|
| Logistic Regression | 0.8536 | 0.7942 | 0.7324 | 0.7877 |
| Random Forest | 0.8435 | 0.8242 | 0.7383 | 0.7821 |
| XGBoost | 0.8206 | 0.7893 | 0.7234 | 0.7821 |
| LightGBM | 0.8030 | 0.7649 | 0.7111 | 0.7821 |

**Best Model**: Random Forest (balanced performance, interpretable feature importance)

## Project Structure

```
phase1_titanic/
├── data/              # titanic_clean.csv
├── notebooks/         # 01-10 numbered notebooks
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── serve.py
├── models/            # titanic_pipeline.joblib
├── api/               # FastAPI app
├── tests/             # pytest tests
├── Dockerfile
├── requirements.txt
└── README.md
```

## Quick Start

### Local Development

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train models
cd src && python train.py

# Run API locally (port 8001)
python api/main.py
# Test: curl -X POST http://localhost:8001/predict -H "Content-Type: application/json" -d '{"pclass": 1, "sex": "female", "age": 25, "sibsp": 0, "parch": 0, "fare": 100, "embarked": "S", "adult_male": false, "deck": "C", "family_size": 1, "fare_per_person": 100, "is_alone": 1, "age_bin": "Young Adult"}'
```

### Docker

```bash
docker build -t titanic-api .
docker run -p 8001:8000 titanic-api
```

### MLflow Tracking

```bash
MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri file:mlruns
# Open http://localhost:5000
```

All experiments logged to `mlruns/`:
- Logistic Regression baseline
- Random Forest (best)
- XGBoost
- LightGBM

Run `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri file:mlruns` to view experiments.

## API Usage

```bash
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "S",
    "adult_male": false,
    "deck": "C",
    "family_size": 1,
    "fare_per_person": 100,
    "is_alone": 1,
    "age_bin": "Young Adult"
  }'
```

Response:
```json
{
  "survived": 1,
  "survival_probability": 0.9598,
  "death_probability": 0.0402,
  "risk_category": "high"
}
```

## MLflow Tracking

All experiments logged to `mlruns/`:
- Logistic Regression baseline
- Random Forest (best)
- XGBoost
- LightGBM

Run `MLFLOW_ALLOW_FILE_STORE=true mlflow ui --backend-store-uri file:mlruns` to view experiments.

## Tests

```bash
MLFLOW_ALLOW_FILE_STORE=true pytest tests/ -v
```

Tests cover:
- No missing values after preprocessing
- Known input → known output shape
- Feature engineering invariants
- Pipeline serialization round-trip
- Class balance preservation in stratified split

## Key Learnings

1. **Pipelines prevent leakage** — ColumnTransformer fit only on training folds
2. **Imbalance handling** — `class_weight='balanced'` works better than SMOTE for tree models
3. **Threshold tuning** — Default 0.5 is rarely optimal; use Youden's J or cost-based thresholds
4. **SHAP for explainability** — Random Forest feature importance aligns with domain knowledge (sex, fare, class)
5. **Data leakage prevention** — Removed leaky columns (`alive`, `class`, `who`, `embark_town`, `alone`) before modeling

## Docker Hub

```bash
docker build -t bakr1m/titanic-survival-prediction .
docker push bakr1m/titanic-survival-prediction
```

Run from Docker Hub:
```bash
docker run -p 8001:8000 bakr1m/titanic-survival-prediction
```