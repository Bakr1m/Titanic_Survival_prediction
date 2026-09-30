"""
Training script for Titanic project
"""
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"

import pandas as pd
import joblib
import mlflow
from sklearn.model_selection import train_test_split

from preprocessing import build_preprocessor


def train_models():
    """Train all models and log to MLflow"""
    # Load and prepare data
    df = pd.read_csv('data/titanic_clean.csv')

    # Drop leaky/redundant columns
    leaky_cols = ['survived', 'alive', 'class', 'who', 'embark_town', 'alone']
    X = df.drop(columns=leaky_cols)
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Build preprocessor
    preprocessor, numeric_features, categorical_features = build_preprocessor(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # MLflow setup
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("titanic_survival")

    # 1. Logistic Regression
    with mlflow.start_run(run_name="logistic_regression"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_param("random_state", 42)

        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
        lr.fit(X_train_processed, y_train)

        y_proba = lr.predict_proba(X_test_processed)[:, 1]
        y_pred = lr.predict(X_test_processed)

        from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(lr, "model")

        print(f"LR: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    # 2. Random Forest
    with mlflow.start_run(run_name="random_forest"):
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 200)
        mlflow.log_param("max_depth", 5)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("random_state", 42)

        from sklearn.ensemble import RandomForestClassifier
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=5, class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train_processed, y_train)

        y_proba = rf.predict_proba(X_test_processed)[:, 1]
        y_pred = rf.predict(X_test_processed)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(rf, "model")

        print(f"RF: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    # 3. XGBoost
    with mlflow.start_run(run_name="xgboost"):
        params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "scale_pos_weight": len(y_train[y_train==0]) / len(y_train[y_train==1]),
            "random_state": 42,
            "eval_metric": "logloss",
            "n_jobs": -1
        }

        import xgboost as xgb
        xgb_clf = xgb.XGBClassifier(**params)
        xgb_clf.fit(X_train_processed, y_train)

        y_proba = xgb_clf.predict_proba(X_test_processed)[:, 1]
        y_pred = xgb_clf.predict(X_test_processed)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        xgb_params = {
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
        }
        for k, v in xgb_params.items():
            mlflow.log_param(k, v)
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.xgboost.log_model(xgb_clf, "model")

        print(f"XGBoost: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    # 4. LightGBM
    with mlflow.start_run(run_name="lightgbm"):
        import lightgbm as lgb
        lgb_clf = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            class_weight='balanced',
            random_state=42,
            verbose=-1,
            n_jobs=-1
        )
        lgb_clf.fit(X_train_processed, y_train)

        y_proba = lgb_clf.predict_proba(X_test_processed)[:, 1]
        y_pred = lgb_clf.predict(X_test_processed)

        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)
        f1 = f1_score(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.lightgbm.log_model(lgb_clf, "model")

        print(f"LightGBM: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, F1={f1:.4f}, Acc={acc:.4f}")

    # Save best pipeline (Random Forest - typically best for Titanic)
    from sklearn.pipeline import Pipeline

    # Refit preprocessor on full dataset (without leaky columns) for final pipeline
    leaky_cols = ['survived', 'alive', 'class', 'who', 'embark_town', 'alone']
    full_df = pd.read_csv('data/titanic_clean.csv')
    X_full = full_df.drop(columns=leaky_cols)

    preprocessor_full, _, _ = build_preprocessor(X_full)
    preprocessor_full.fit(X_full)

    best_pipeline = Pipeline([
        ('preprocessor', preprocessor_full),
        ('classifier', rf)
    ])

    joblib.dump(best_pipeline, 'models/titanic_pipeline.joblib')
    print("Best pipeline saved to models/titanic_pipeline.joblib")


if __name__ == "__main__":
    train_models()
