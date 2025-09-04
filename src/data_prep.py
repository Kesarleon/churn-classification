import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def load_data(path="data/churn_data.csv"):
    return pd.read_csv(path)

def create_preprocessor():
    numeric_features = ["age", "tenure", "monthly_charges"]
    categorical_features = ["contract_type", "internet_service"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    return preprocessor

def split_data(df):
    X = df.drop(columns=["churn", "customer_id"])
    y = df["churn"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
