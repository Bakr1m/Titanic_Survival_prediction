"""
Preprocessing pipeline for Titanic project
"""
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer


def load_raw_data():
    """Load raw Titanic data from seaborn"""
    import seaborn as sns
    df = sns.load_dataset('titanic')
    return df


def clean_data(df):
    """Apply cleaning steps to raw Titanic data"""
    df = df.copy()

    # --- age: ~20% missing ---
    # Strategy: median imputation by sex + pclass
    age_median = df.groupby(['sex', 'pclass'])['age'].transform('median')
    df['age'] = df['age'].fillna(age_median)

    # --- embarked: 2 missing ---
    df['embarked'] = df['embarked'].fillna(df['embarked'].mode()[0])
    df['embark_town'] = df['embark_town'].fillna(df['embark_town'].mode()[0])

    # --- deck: 77% missing ---
    df['deck'] = df['deck'].cat.add_categories('Unknown').fillna('Unknown')

    return df


def engineer_features(df):
    """Add engineered features"""
    df = df.copy()

    # Family size
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    # Is alone
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    # Fare per person
    df['fare_per_person'] = df['fare'] / df['family_size']

    # Age bins
    df['age_bin'] = pd.cut(df['age'],
                           bins=[0, 12, 18, 35, 60, 80],
                           labels=['Child', 'Teen', 'Young Adult', 'Adult', 'Senior'])

    return df


def build_preprocessor(X_train):
    """Build and return ColumnTransformer with numeric and categorical pipelines"""
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

    return preprocessor, numeric_features, categorical_features


def prepare_data():
    """Full data preparation pipeline"""
    # Load and clean
    df = load_raw_data()
    df = clean_data(df)
    df = engineer_features(df)

    # Save cleaned data
    df.to_csv('data/titanic_clean.csv', index=False)

    return df


def get_features_target(df):
    """Split dataframe into features and target"""
    # Drop leaky/redundant columns
    leaky_cols = ['survived', 'alive', 'class', 'who', 'embark_town', 'alone']
    feature_cols = [c for c in df.columns if c not in leaky_cols]
    X = df[feature_cols]
    y = df['survived']
    return X, y
