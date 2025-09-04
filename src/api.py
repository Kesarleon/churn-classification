from fastapi import FastAPI
from pydantic import BaseModel
from .predict import predict_single

app = FastAPI(title="Churn Prediction API")

class Customer(BaseModel):
    age: int
    tenure: int
    monthly_charges: float
    contract_type: str
    internet_service: str

@app.post("/predict")
def predict(customer: Customer):
    data = customer.dict()
    result = predict_single(data)
    return {"churn_prediction": result}
