import pandas as pd
import numpy as np

np.random.seed(42)

n = 1000
data = pd.DataFrame({
    "customer_id": range(1, n+1),
    "age": np.random.randint(18, 70, n),
    "tenure": np.random.randint(1, 60, n),  # meses como cliente
    "monthly_charges": np.random.uniform(20, 120, n).round(2),
    "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.25, 0.15]),
    "internet_service": np.random.choice(["DSL", "Fiber optic", "None"], n),
    "churn": np.random.choice([0, 1], n, p=[0.73, 0.27])  # 27% churn rate
})

data.to_csv("data/churn_data.csv", index=False)
print("Dataset generado en data/churn_data.csv")
