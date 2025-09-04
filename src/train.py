import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from .data_prep import load_data, split_data, create_preprocessor

def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = create_preprocessor()

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "models/churn_model.pkl")
    print("Modelo guardado en models/churn_model.pkl")

if __name__ == "__main__":
    train_model()
