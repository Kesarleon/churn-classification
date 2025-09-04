import pandas as pd
import numpy as np
import logging
from . import config

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_data():
    logging.info("Generando dataset sintético de churn...")
    np.random.seed(config.RANDOM_STATE)
    n = 1000
    data = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 70, n),
        "tenure": np.random.randint(1, 60, n),
        "monthly_charges": np.random.uniform(20, 120, n).round(2),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.25, 0.15]),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "None"], n),
        "churn": np.random.choice([0, 1], n, p=[0.73, 0.27])
    })

    data.to_csv(config.DATA_PATH, index=False)
    logging.info(f"Dataset generado y guardado en {config.DATA_PATH}")

if __name__ == "__main__":
    generate_data()
