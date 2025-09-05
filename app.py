import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import sys
sys.path.append('src')

from config import MODEL_PATH, DATA_PATH, FEATURES

# --------------------------
# Configuración inicial
# --------------------------
st.set_page_config(
    page_title="Churn Classification Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction Dashboard")
st.markdown("Interactúa con el modelo de churn y explora datos de clientes.")

# --------------------------
# Cargar modelo y dataset
# --------------------------
@st.cache_resource
def load_model():
    try:
        return joblib.load(MODEL_PATH)
    except FileNotFoundError:
        st.error(f"No se encontró el modelo en la ruta: {MODEL_PATH}. Asegúrate de entrenar el modelo primero con `python -m src.train`.")
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        st.error(f"No se encontró el dataset en la ruta: {DATA_PATH}. Asegúrate de generar los datos primero con `python -m src.generate_data`.")
        return None

model = load_model()
df = load_data()

# --------------------------
# Sección 1: Input de usuario para predicción
# --------------------------
st.sidebar.header("🔧 Simulación de Cliente para Predicción")

age = st.sidebar.slider("Edad (age)", 18, 100, 42)
tenure = st.sidebar.slider("Meses como cliente (tenure)", 0, 72, 8)
monthly_charges = st.sidebar.slider("Cargos mensuales", 0, 150, 85)
contract_type = st.sidebar.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "None"])

if st.sidebar.button("Predecir churn"):
    if model is not None:
        # Crear DataFrame con el orden de columnas correcto
        input_data = pd.DataFrame({
            "age": [age],
            "tenure": [tenure],
            "monthly_charges": [monthly_charges],
            "contract_type": [contract_type],
            "internet_service": [internet_service]
        })
        input_data = input_data[FEATURES] # Reordenar según la lista de features del config

        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        st.subheader("🔮 Resultado de la Predicción")
        if prediction == 1:
            st.warning(f"**Predicción de Churn:** SÍ (Probabilidad: {prob:.2%})")
        else:
            st.success(f"**Predicción de Churn:** NO (Probabilidad de Churn: {prob:.2%})")

# --------------------------
# Sección 2: Exploración de datos
# --------------------------
if df is not None:
    st.subheader("📈 Exploración de Datos de Clientes")

    col1, col2 = st.columns(2)

    with col1:
        st.write("#### Distribución de Antigüedad (tenure) por Churn")
        fig1 = px.histogram(df, x="tenure", color="churn", barmode="overlay",
                                title="Distribución de Antigüedad vs. Churn",
                                labels={"churn": "Churn", "tenure": "Antigüedad (meses)"})
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.write("#### Cargos Mensuales por Churn")
        fig2 = px.box(df, x="churn", y="monthly_charges", color="churn",
                          title="Cargos Mensuales vs. Churn",
                          labels={"churn": "Churn", "monthly_charges": "Cargos Mensuales"})
        st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No se pueden mostrar las visualizaciones porque el dataset no se ha cargado.")
