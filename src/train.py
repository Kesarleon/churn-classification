import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    logging.info("--- Inicio del pipeline de entrenamiento y competición de modelos ---")

    # 1. Cargar y preparar datos
    logging.info("Paso 1: Cargando y preparando datos...")
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    preprocessor = create_preprocessor()
    logging.info("Datos listos para el entrenamiento.")

    best_model = None
    best_metric = -1
    best_model_name = ""
    best_model_metrics = {}

    # 2. Competición de modelos
    logging.info("Paso 2: Iniciando competición de modelos...")
    for name, model in config.MODELS.items():
        logging.info(f"--- Entrenando y evaluando: {name} ---")

        # Crear pipeline para el modelo actual
        pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ])

        # Entrenar
        pipeline.fit(X_train, y_train)

        # Evaluar
        y_pred = pipeline.predict(X_test)
        metric_value = f1_score(y_test, y_pred) # Usando F1 como métrica principal

        logging.info(f"Modelo: {name}, F1-Score: {metric_value:.4f}")

        # Guardar el mejor modelo
        if metric_value > best_metric:
            best_metric = metric_value
            best_model = pipeline
            best_model_name = name

            # Guardar todas las métricas del mejor modelo
            cm = confusion_matrix(y_test, y_pred)
            best_model_metrics = {
                "model_name": name,
                "accuracy": accuracy_score(y_test, y_pred),
                "f1_score": f1_score(y_test, y_pred),
                "confusion_matrix": cm.tolist(),
                "report": classification_report(y_test, y_pred, output_dict=True)
            }
            logging.info(f"¡Nuevo mejor modelo encontrado!: {name} con F1-Score de {best_metric:.4f}")

    logging.info(f"--- Competición finalizada. Mejor modelo: {best_model_name} ---")

    # 3. Guardar el mejor modelo y sus artefactos
    if best_model:
        logging.info("Paso 3: Guardando el mejor modelo y sus artefactos...")
        joblib.dump(best_model, config.MODEL_PATH)
        save_metadata(best_model_metrics, config.METADATA_PATH)

        # Re-generar y_pred del mejor modelo para la matriz de confusión
        y_pred_best = best_model.predict(X_test)
        save_confusion_matrix_plot(y_test, y_pred_best, config.CONFUSION_MATRIX_PATH)

        logging.info("Artefactos del mejor modelo guardados correctamente.")
    else:
        logging.warning("No se encontró ningún modelo para guardar.")

    logging.info("--- Fin del pipeline ---")

if __name__ == "__main__":
    train_model()
