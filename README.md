# Titanic Survival Prediction

**Foundations capstone | ML Lifecycle: data → model → evaluation → API → Docker**

## Business Context

Predict survival on the Titanic using passenger demographics and ticket
information. A foundations "dress rehearsal" for the healthcare portfolio:
the same engineering bar (pipelines, hermetic tests, CI, release-pinned
artifacts, Docker) on a small, fully understood dataset.

## Dataset

- **Source**: seaborn built-in `titanic` (Kaggle Titanic, 891 passengers);
  reproduced locally via `scripts/download_data.sh` (data/ is gitignored)
- **Target**: `survived` (binary; 61.6% died, 38.4% survived)
- **Features**: 13 leak-free columns (demographics, ticket info, engineered
  `family_size`, `is_alone`, `fare_per_person`, `age_bin`)
- **Excluded as leaky**: `alive`, `class`, `who`, `embark_town`, `alone`
  (label-derived or redundant) — the API rejects them with 422

## Approach

1. **EDA + cleaning** (`notebooks/02–03`): age median-by-sex×class, deck →
   `Unknown`, 2 missing embarked.
2. **Feature engineering** (`src/preprocessing.py`): family size, alone flag,
   fare-per-person, age bins.
3. **Pipelines** (`src/train.py`): ColumnTransformer fit on train only
   (median/most-frequent → scale/one-hot); stratified 80/20 split.
4. **Baselines** (`notebooks/05–09`): LR, RF, XGBoost, LightGBM + PyTorch;
   imbalance via `class_weight='balanced'`; MLflow tracking.
5. **Serving** (`src/serve.py`): FastAPI `/predict`, lazy model load,
   feature order derived from the fitted transformer.
6. **Container**: release-pinned artifact, SHA256-verified, parity-checked.

## Results

| Model | ROC-AUC | PR-AUC | F1 | Accuracy |
|-------|---------|--------|-----|----------|
| Logistic Regression | 0.8536 | 0.7942 | 0.7324 | 0.7877 |
| Random Forest | 0.8435 | 0.8242 | 0.7383 | 0.7821 |
| XGBoost | 0.8206 | 0.7893 | 0.7234 | 0.7821 |
| LightGBM | 0.8030 | 0.7649 | 0.7111 | 0.7821 |

**Shipped model**: Random Forest (balanced performance, interpretable
importance: sex, fare, class). Example: 1st-class young woman → survival
probability 0.9598.

## Limitations

1. **Toy 1912 data**: no generalization claim to any modern population.
2. **`deck` 77% missing** (imputed `Unknown`); `age` ~20% missing.
3. **Default 0.5 threshold**: no calibration or cost tuning.
4. **Single 80/20 split**: no cross-validation; small-sample variance.
5. **Historical bias**: importance reflects 1912 evacuation patterns
   ("women and children first") — descriptive, not normative.

## Ethical Considerations

Demo only. The model must never inform any real decision; its strongest
signals (sex, class, fare) encode historical inequity, and presenting them
as predictors without that context would be misleading.

## Run Instructions

```bash
make install-dev   # serving deps + pytest/ruff/httpx
make test lint     # 11 tests, ruff clean
make install-train # + xgboost/lightgbm/mlflow/shap/seaborn
bash scripts/download_data.sh  # rebuild data/titanic_clean.csv
make train         # retrain; saves models/titanic_pipeline.joblib
make serve         # API on :8000
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d @example_passenger.json
```

Docker (prebuilt, release-pinned artifact inside):

```bash
docker pull bakr1m/titanic-api:latest
docker run -p 8000:8000 bakr1m/titanic-api:latest
```

## Key Learnings

1. **Pipelines prevent leakage** — ColumnTransformer fit only on training folds.
2. **`class_weight='balanced'`** beats SMOTE for tree models on this data.
3. **Leakage is a contract problem** — the first serving schema required the
   label-derived columns the model excluded; now rejected with 422.
4. **Hermetic tests** — synthetic same-schema fallback keeps CI green with
   no `data/` present.
5. **Default 0.5 is rarely optimal** — Youden's J or cost-based thresholds.
