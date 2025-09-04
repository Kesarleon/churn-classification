import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix

from . import config
from .data_prep import load_data, split_data, create_preprocessor
from .utils import save_metadata

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_confusion_matrix_plot(y_true, y_pred, path):
    """Genera y guarda la matriz de confusión como una imagen."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.xlabel('Predicción')
    plt.ylabel('Verdadero')
    plt.title('Matriz de Confusión')
    plt.savefig(path)
    plt.close()
    logging.info(f"Matriz de confusión guardada en {path}")

def train_model():
    logging.info("--- Inicio del pipeline de entrenamiento ---")

    # 1. Cargar datos
    logging.info("Paso 1: Cargando datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    logging.info(f"Datos cargados: {len(X_train)} muestras de entrenamiento, {len(X_test)} muestras de prueba.")

    # 2. Crear pipeline
    logging.info("Paso 2: Creando el pipeline de preprocesamiento y modelo...")
    preprocessor = create_preprocessor()
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(
            max_iter=config.MAX_ITER,
            class_weight='balanced'
        ))
    ])

    # 3. Entrenar
    logging.info("Paso 3: Entrenando el modelo...")
    model.fit(X_train, y_train)
    logging.info("Entrenamiento completado.")

    # 4. Evaluar
    logging.info("Paso 4: Evaluando el modelo...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "confusion_matrix": cm.tolist(), # Convertir a lista para que sea serializable en JSON
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    logging.info("Reporte de clasificación:\n" + classification_report(y_test, y_pred))

    # 5. Guardar modelo, metadata y visualizaciones
    logging.info("Paso 5: Guardando artefactos del modelo...")
    joblib.dump(model, config.MODEL_PATH)
    save_metadata(metrics, config.METADATA_PATH)
    save_confusion_matrix_plot(y_test, y_pred, config.CONFUSION_MATRIX_PATH)

    logging.info("--- Fin del pipeline de entrenamiento ---")

if __name__ == "__main__":
    train_model()
