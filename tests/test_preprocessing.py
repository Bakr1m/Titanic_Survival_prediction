"""
Pytest tests for Titanic preprocessing pipeline.

Hermetic: uses the real data/titanic_clean.csv when present, otherwise
falls back to synthetic same-schema rows — CI has no data/ (gitignored).
"""
import pytest
import pandas as pd
import numpy as np
import joblib
import tempfile
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _synthetic_rows(n=200, seed=42):
    """Same-schema stand-in for data/titanic_clean.csv (CI has no data/)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "survived": rng.integers(0, 2, n),
        "pclass": rng.integers(1, 4, n),
        "sex": rng.choice(["male", "female"], n),
        "age": rng.uniform(1, 80, n).round(1),
        "sibsp": rng.integers(0, 4, n),
        "parch": rng.integers(0, 3, n),
        "fare": rng.uniform(5, 200, n).round(2),
        "embarked": rng.choice(["S", "C", "Q"], n),
        "adult_male": rng.choice([True, False], n),
        "deck": rng.choice(["A", "B", "C", "Unknown"], n),
        "family_size": rng.integers(1, 5, n),
        "is_alone": rng.integers(0, 2, n),
        "fare_per_person": rng.uniform(2, 100, n).round(2),
        "age_bin": rng.choice(["Child", "Teen", "Young Adult", "Adult", "Senior"], n),
        "alive": rng.choice(["yes", "no"], n),
        "class": rng.choice(["First", "Second", "Third"], n),
        "who": rng.choice(["man", "woman", "child"], n),
        "embark_town": rng.choice(["Southampton", "Cherbourg", "Queenstown"], n),
        "alone": rng.choice([True, False], n),
    })


# Load data and build preprocessor (shared fixture)
@pytest.fixture(scope="module")
def data_and_preprocessor():
    csv = PROJECT_ROOT / "data" / "titanic_clean.csv"
    df = pd.read_csv(csv) if csv.exists() else _synthetic_rows()

    leaky_cols = ['survived', 'alive', 'class', 'who', 'embark_town', 'alone']
    X = df.drop(columns=leaky_cols)
    y = df['survived']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    numeric_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    return {
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'preprocessor': preprocessor,
        'X_train_processed': X_train_processed,
        'X_test_processed': X_test_processed,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }


def test_no_missing_after_transform(data_and_preprocessor):
    """Test that preprocessor output has no missing values"""
    X_test_proc = data_and_preprocessor['X_test_processed']
    if hasattr(X_test_proc, 'toarray'):
        X_test_proc = X_test_proc.toarray()
    assert not np.isnan(X_test_proc).any(), "Preprocessor output contains NaN values"


def test_known_input_known_output(data_and_preprocessor):
    """Test specific known input produces expected output shape and no errors"""
    preprocessor = data_and_preprocessor['preprocessor']

    known_input = data_and_preprocessor['X_train'].iloc[:1]
    transformed = preprocessor.transform(known_input)

    expected_n_features = data_and_preprocessor['X_train_processed'].shape[1]
    if hasattr(transformed, 'toarray'):
        transformed = transformed.toarray()
    assert transformed.shape == (1, expected_n_features), \
        f"Expected shape (1, {expected_n_features}), got {transformed.shape}"

    assert not np.isnan(transformed).any(), "Known input produced NaN output"


def test_feature_engineering_invariants(data_and_preprocessor):
    """Test that engineered features maintain expected invariants"""
    X_train = data_and_preprocessor['X_train']

    assert (X_train['family_size'] >= 1).all(), "family_size has invalid values"
    assert (X_train['is_alone'].isin([0, 1])).all(), "is_alone not binary"
    assert (X_train['fare_per_person'] >= 0).all(), "fare_per_person negative"

    valid_age_bins = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
    assert X_train['age_bin'].isin(valid_age_bins).all(), "Invalid age_bin values"
    # age_midpoint was dropped as leaky column


def test_pipeline_serialization(data_and_preprocessor):
    """Test that preprocessor can be saved and loaded correctly"""
    preprocessor = data_and_preprocessor['preprocessor']

    with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
        temp_path = f.name

    try:
        joblib.dump(preprocessor, temp_path)
        loaded_preprocessor = joblib.load(temp_path)

        test_input = data_and_preprocessor['X_train'].iloc[:5]
        original_output = preprocessor.transform(test_input)
        loaded_output = loaded_preprocessor.transform(test_input)

        if hasattr(original_output, 'toarray'):
            original_output = original_output.toarray()
        if hasattr(loaded_output, 'toarray'):
            loaded_output = loaded_output.toarray()

        np.testing.assert_array_almost_equal(original_output, loaded_output, decimal=5)
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_class_balance_preserved(data_and_preprocessor):
    """Test that stratified split preserves class balance"""
    y_train = data_and_preprocessor['y_train']
    y_test = data_and_preprocessor['y_test']

    train_prop = data_and_preprocessor['y_train'].mean()
    test_prop = data_and_preprocessor['y_test'].mean()
    overall_prop = pd.concat([y_train, y_test]).mean()

    assert abs(train_prop - overall_prop) < 0.02, (
        f"Train prop {train_prop:.4f} vs overall {overall_prop:.4f}"
    )
    assert abs(test_prop - overall_prop) < 0.02, (
        f"Test prop {test_prop:.4f} vs overall {overall_prop:.4f}"
    )
