# Model Card — Titanic Survival Classifier

## Intended use
Teaching/demonstration of the full ML lifecycle (EDA → pipeline → API →
Docker) on the Kaggle Titanic dataset. Not for any real-world decision.

## Data
- **Source**: seaborn built-in `titanic` (Kaggle Titanic, 891 passengers).
- **Target**: `survived` (0/1; 61.6% died, 38.4% survived).
- **Features**: 13 leak-free columns (demographics, ticket, engineered
  `family_size`, `is_alone`, `fare_per_person`, `age_bin`).
- **Excluded**: `alive`, `class`, `who`, `embark_town`, `alone` — derived
  from or redundant with the label; the API rejects them with 422.

## Model
- Random Forest (200 trees, max_depth 5, class_weight balanced), wrapped in
  a sklearn Pipeline with the fitted ColumnTransformer (median/most-frequent
  imputation, standard scaling, one-hot with unknown-ignore).
- Compared against Logistic Regression (ROC-AUC 0.8536), XGBoost (0.8206),
  LightGBM (0.8030); RF chosen for balanced performance + interpretability.
- Train/test: stratified 80/20 split, random_state 42.

## Metrics (held-out test)
| Model | ROC-AUC | PR-AUC | F1 | Accuracy |
|-------|---------|--------|-----|----------|
| Logistic Regression | 0.8536 | 0.7942 | 0.7324 | 0.7877 |
| Random Forest | 0.8435 | 0.8242 | 0.7383 | 0.7821 |
| XGBoost | 0.8206 | 0.7893 | 0.7234 | 0.7821 |
| LightGBM | 0.8030 | 0.7649 | 0.7111 | 0.7821 |

## Limitations
- Toy dataset from 1912; no generalization claim to any modern population.
- `deck` is 77% missing (imputed as `Unknown`); `age` ~20% missing
  (median by sex × class).
- Default 0.5 threshold; no calibration or cost tuning.
- Single 80/20 split — no cross-validation.

## Ethics
Historical data with class/gender structure; feature importance reflects
1912 evacuation patterns ("women and children first"), not a normative rule.
Demo only.
