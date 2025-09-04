import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from . import config

def load_data(path=config.DATA_PATH):
    return pd.read_csv(path)

def create_preprocessor():
    # Pipeline para transformar variables numéricas: imputar y escalar
    numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para transformar variables categóricas: imputar y codificar
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # ColumnTransformer para aplicar las transformaciones correctas a cada tipo de columna
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_pipeline, config.NUMERIC_FEATURES),
            ('cat', categorical_pipeline, config.CATEGORICAL_FEATURES)
        ],
        remainder='passthrough'
    )
    return preprocessor

def split_data(df):
    X = df.drop(columns=[config.TARGET, config.CUSTOMER_ID])
    y = df[config.TARGET]
    return train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )
