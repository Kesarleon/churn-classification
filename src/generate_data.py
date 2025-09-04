import pandas as pd
import numpy as np
import logging
from . import config

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def generate_data():
    logging.info("Generando dataset sintético de churn más realista...")
    np.random.seed(config.RANDOM_STATE)
    n = 1000

    # 1. Generar datos base
    data = pd.DataFrame({
        "customer_id": range(1, n + 1),
        "age": np.random.randint(18, 70, n),
        "tenure": np.random.randint(1, 60, n),
        "monthly_charges": np.random.uniform(20, 120, n).round(2),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], n, p=[0.6, 0.25, 0.15]),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "None"], n, p=[0.4, 0.5, 0.1]),
        # 2. Aumentar desbalanceo de clase (e.g., 15% churn)
        "churn": np.random.choice([0, 1], n, p=[0.85, 0.15])
    })

    # 3. Introducir datos faltantes
    # 5% de valores faltantes en 'age' y 'monthly_charges'
    missing_indices_age = data.sample(frac=0.05, random_state=config.RANDOM_STATE).index
    data.loc[missing_indices_age, 'age'] = np.nan

    missing_indices_charges = data.sample(frac=0.05, random_state=config.RANDOM_STATE + 1).index
    data.loc[missing_indices_charges, 'monthly_charges'] = np.nan

    # 5% de valores "Unknown" en 'internet_service'
    missing_indices_service = data.sample(frac=0.05, random_state=config.RANDOM_STATE + 2).index
    data.loc[missing_indices_service, 'internet_service'] = 'Unknown'

    logging.info(f"Datos generados con {data.isnull().sum().sum()} valores nulos.")

    data.to_csv(config.DATA_PATH, index=False)
    logging.info(f"Dataset generado y guardado en {config.DATA_PATH}")

if __name__ == "__main__":
    generate_data()
