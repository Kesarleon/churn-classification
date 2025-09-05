import streamlit as st
import json
from PIL import Image
import pandas as pd
import sys
sys.path.append('src')

from config import METADATA_PATH, CONFUSION_MATRIX_PATH

st.set_page_config(page_title="Dashboard de Desempeño del Modelo de Churn", layout="wide")

st.title("📊 Dashboard de Desempeño del Modelo de Churn")
st.write("Este dashboard muestra las métricas de rendimiento del **mejor modelo** seleccionado tras una competición.")

try:
    # --- Cargar Artefactos ---
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)

    confusion_matrix_img = Image.open(CONFUSION_MATRIX_PATH)

    # --- Mostrar Información del Mejor Modelo ---
    st.header(f"🏆 Mejor Modelo: `{metadata['metrics']['model_name']}`")

    # --- Mostrar Métricas ---
    st.subheader("Métricas Clave")
    col1, col2 = st.columns(2)
    col1.metric("Accuracy Score", f"{metadata['metrics']['accuracy']:.4f}")
    col2.metric("F1 Score (Ponderado)", f"{metadata['metrics']['f1_score']:.4f}")

    st.subheader("Matriz de Confusión")
    st.image(confusion_matrix_img, caption="Matriz de Confusión del conjunto de prueba.")

    st.subheader("Reporte de Clasificación")
    report_df = pd.DataFrame(metadata['metrics']['report']).transpose()
    st.dataframe(report_df)

    st.sidebar.info(f"Último entrenamiento: {metadata['created_at']}")

except FileNotFoundError:
    st.error("No se encontraron los artefactos del modelo (metadata.json o confusion_matrix.png).")
    st.warning("Por favor, ejecuta el pipeline de entrenamiento primero con `python -m src.train`.")
