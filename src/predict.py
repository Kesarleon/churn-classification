import joblib
import pandas as pd

MODEL_PATH = "models/churn_model.pkl"

def load_model():
    return joblib.load(MODEL_PATH)

def predict_single(data_dict):
    model = load_model()
    df = pd.DataFrame([data_dict])
    return int(model.predict(df)[0])

def predict_batch(csv_path):
    model = load_model()
    df = pd.read_csv(csv_path)
    preds = model.predict(df)
    df["churn_prediction"] = preds
    return df

if __name__ == "__main__":
    sample = {
        "age": 42,
        "tenure": 8,
        "monthly_charges": 85,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic"
    }
    print("Predicción cliente:", predict_single(sample))
