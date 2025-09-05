import joblib
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from . import config
from .data_prep import create_preprocessor
from .utils import save_metadata

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def create_dummy_artifacts():
    """
    Crea artefactos de modelo 'dummy' para fines de demostración.
    Entrena un modelo simple con 10 filas de datos para asegurar que los artefactos
    sean válidos y puedan ser cargados por los dashboards.
    """
    logging.info("--- Creando artefactos dummy para el demo ---")

    # 1. Crear un pequeño dataset dummy
    dummy_data = {
        "age": np.random.randint(18, 70, 10),
        "tenure": np.random.randint(1, 60, 10),
        "monthly_charges": np.random.uniform(20, 120, 10).round(2),
        "contract_type": np.random.choice(["Month-to-month", "One year", "Two year"], 10),
        "internet_service": np.random.choice(["DSL", "Fiber optic", "None"], 10),
        "churn": np.random.choice([0, 1], 10)
    }
    df = pd.DataFrame(dummy_data)
    X = df[config.FEATURES]
    y = df[config.TARGET]

    # 2. Crear y entrenar un pipeline dummy
    logging.info("Entrenando un modelo dummy con 10 muestras...")
    preprocessor = create_preprocessor()
    dummy_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression()) # Un clasificador simple
    ])
    dummy_model.fit(X, y)
    logging.info("Modelo dummy entrenado.")

    # 3. Guardar el modelo dummy
    joblib.dump(dummy_model, config.MODEL_PATH)
    logging.info(f"Modelo dummy guardado en {config.MODEL_PATH}")

    # 4. Crear y guardar metadata dummy
    dummy_metrics = {
        "model_name": "DummyModel",
        "accuracy": 0.5,
        "f1_score": 0.5,
        "confusion_matrix": [[3, 2], [2, 3]],
        "report": {
            "0": {"precision": 0.5, "recall": 0.6, "f1-score": 0.55, "support": 5},
            "1": {"precision": 0.5, "recall": 0.4, "f1-score": 0.45, "support": 5}
        }
    }
    save_metadata(dummy_metrics, config.METADATA_PATH)

    # 5. Crear y guardar una matriz de confusión dummy
    cm = np.array([[3, 2], [2, 3]])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión (Dummy)')
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.close()
    logging.info(f"Matriz de confusión dummy guardada en {config.CONFUSION_MATRIX_PATH}")

    logging.info("--- Artefactos dummy creados exitosamente ---")

if __name__ == "__main__":
    create_dummy_artifacts()
