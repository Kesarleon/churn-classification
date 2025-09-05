# Archivo de Configuración para el Pipeline de Churn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# --- Rutas de Archivos ---
DATA_PATH = "data/churn_data.csv"
MODEL_PATH = "models/churn_model.pkl"
METADATA_PATH = "models/metadata.json"
CONFUSION_MATRIX_PATH = "models/confusion_matrix.png"

# --- Variables del Modelo ---
TARGET = "churn"
CUSTOMER_ID = "customer_id"
NUMERIC_FEATURES = ["age", "tenure", "monthly_charges"]
CATEGORICAL_FEATURES = ["contract_type", "internet_service"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Parámetros de División de Datos ---
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Configuración de la Competición de Modelos ---
# Métrica principal para la selección del modelo
PRIMARY_METRIC = "f1_score"

# Modelos a competir
MODELS = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        class_weight='balanced',
        random_state=RANDOM_STATE
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=RANDOM_STATE
    )
}
