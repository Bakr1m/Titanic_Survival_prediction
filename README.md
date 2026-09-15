# Phase 1: Titanic Survival Prediction — Mini Project

**Days 1–10 | Crash Course Foundations**

This is the Phase 1 capstone mini-project from the 60-Day ML Engineer Roadmap. It demonstrates the complete ML lifecycle: data profiling → cleaning → EDA → modeling → evaluation → API → Docker deployment.

## Business Context

Predict survival on the Titanic using passenger demographics and ticket information. While not a healthcare problem, this serves as the "dress rehearsal" for the 5 real healthcare projects that follow.

## Dataset

- **Source**: Seaborn's built-in Titanic dataset (891 passengers)
- **Target**: `survived` (binary: 0=died, 1=survived)
- **Features**: 17 features after engineering (demographics, ticket info, engineered features)
- **Class Balance**: 61.6% died, 38.4% survived

## Key Results

| Model | ROC-AUC | PR-AUC | F1 | Accuracy |
|-------|---------|--------|-----|----------|
| Logistic Regression | ~0.85 | ~0.80 | ~0.77 | ~0.78 |
| Random Forest | ~0.85 | ~0.83 | ~0.78 | ~0.79 |
| XGBoost | ~0.82 | ~0.24 | ~0.28 | ~0.80 |
| LightGBM | ~0.83 | ~0.24 | ~0.28 | ~0.79 |

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
# Create conda environment
conda create -n titanic python=3.10
conda activate titanic
pip install -r requirements.txt

# Train models
cd src && python train.py

# Run API locally
python api/main.py
# Test: curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"pclass": 1, "sex": "female", "age": 25, ...}'
```

### Docker

```bash
docker build -t titanic-api .
docker run -p 8000:8000 titanic-api
```

### MLflow Tracking

```bash
mlflow ui --backend-store-uri file:mlruns
# Open http://localhost:5000
```

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pclass": 1,
    "sex": "female",
    "age": 25,
    "sibsp": 0,
    "parch": 0,
    "fare": 100,
    "embarked": "S",
    "class": "First",
    "who": "woman",
    "adult_male": false,
    "deck": "C",
    "embark_town": "Southampton",
    "alive": "yes",
    "alone": true,
    "family_size": 1,
    "is_alone": 1,
    "fare_per_person": 100,
    "age_bin": "Young Adult"
  }'
```

Response:
```json
{
  "survived": 1,
  "survival_probability": 0.94,
  "death_probability": 0.06,
  "risk_category": "high"
}
```

## MLflow Tracking

All experiments logged to `mlruns/`:
- Logistic Regression baseline
- Random Forest (best)
- XGBoost
- LightGBM

Run `mlflow ui --backend-store-uri file:mlruns` to view experiments.

## Tests

```bash
pytest tests/ -v
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

## Next Steps

This mini-project serves as the template for 5 healthcare projects:
1. Hospital Readmission Risk Predictor
2. Sepsis Early-Warning System
3. Chest X-Ray Pneumonia Detector
4. Clinical Document RAG Assistant
5. Patient No-Show & Hospital Ops Optimizer