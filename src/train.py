import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score
from .data_prep import load_data, split_data, create_preprocessor
from .utils import save_metadata

def train_model():
    # 1. Cargar datos
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Crear pipeline
    preprocessor = create_preprocessor()
    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    # 3. Entrenar
    model.fit(X_train, y_train)

    # 4. Evaluar
    y_pred = model.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True)
    }
    print(classification_report(y_test, y_pred))

    # 5. Guardar modelo + metadata
    joblib.dump(model, "models/churn_model.pkl")
    save_metadata(metrics, "models/metadata.json")
    print("Modelo y metadata guardados en carpeta models/")

if __name__ == "__main__":
    train_model()
