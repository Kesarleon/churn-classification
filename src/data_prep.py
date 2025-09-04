import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from . import config

def load_data(path=config.DATA_PATH):
    return pd.read_csv(path)

def create_preprocessor():
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, config.NUMERIC_FEATURES),
            ("cat", categorical_transformer, config.CATEGORICAL_FEATURES),
        ],
        remainder="passthrough"  # Keep other columns if any
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
