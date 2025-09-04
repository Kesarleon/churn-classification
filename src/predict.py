import joblib
import pandas as pd

def predict_new(data_dict):
    model = joblib.load("models/churn_model.pkl")
    df = pd.DataFrame([data_dict])
    return model.predict(df)[0]

if __name__ == "__main__":
    sample = {
        "age": 35,
        "tenure": 12,
        "monthly_charges": 70,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic"
    }
    print("Predicción churn:", predict_new(sample))
