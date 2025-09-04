import joblib
import pandas as pd

from . import config

def load_model():
    return joblib.load(config.MODEL_PATH)

def predict_single(data_dict):
    model = load_model()
    # Asegurar que el dataframe tenga las columnas en el orden correcto
    df = pd.DataFrame([data_dict])
    df = df[config.FEATURES]
    return int(model.predict(df)[0])

def predict_batch(csv_path):
    model = load_model()
    df = pd.read_csv(csv_path)
    # Asegurar que el dataframe tenga las columnas en el orden correcto
    X = df[config.FEATURES]
    preds = model.predict(X)
    df["churn_prediction"] = preds
    return df

if __name__ == "__main__":
    # Ejemplo de un cliente para predicción
    sample = {
        "age": 42,
        "tenure": 8,
        "monthly_charges": 85.0,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic"
    }
    print("Predicción cliente:", predict_single(sample))
