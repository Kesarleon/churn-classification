# Archivo de Configuración para el Pipeline de Churn

# --- Rutas de Archivos ---
DATA_PATH = "data/churn_data.csv"
MODEL_PATH = "models/churn_model.pkl"
METADATA_PATH = "models/metadata.json"

# --- Variables del Modelo ---
TARGET = "churn"
CUSTOMER_ID = "customer_id"
NUMERIC_FEATURES = ["age", "tenure", "monthly_charges"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Parámetros de División de Datos ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Parámetros del Modelo ---
MAX_ITER = 1000
