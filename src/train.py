import joblib
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

from . import config
from .data_prep import load_data, split_data, create_preprocessor
from .utils import save_metadata

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    logging.info("Reporte de clasificación:\n" + classification_report(y_test, y_pred))

    # 5. Guardar modelo + metadata
    logging.info("Paso 5: Guardando el modelo y la metadata...")
    joblib.dump(model, config.MODEL_PATH)
    save_metadata(metrics, config.METADATA_PATH)
    logging.info(f"Modelo y metadata guardados en la carpeta /models.")

    logging.info("--- Fin del pipeline de entrenamiento ---")

if __name__ == "__main__":
    train_model()
