# Modelo de Predicción de Churn de Clientes

Este repositorio contiene un pipeline completo de machine learning para la predicción de churn (fuga de clientes). El proyecto está construido en Python y demuestra un flujo de trabajo de productivización, desde la generación de datos hasta la exposición del modelo a través de una API.

## 🚀 Estructura del Proyecto

El repositorio está organizado de la siguiente manera:

```
├── data/
│   └── churn_data.csv        # Dataset generado (no versionado por defecto en un entorno real)
├── models/
│   ├── churn_model.pkl       # Modelo entrenado (ignorado por .gitignore)
│   └── metadata.json         # Métricas del modelo (ignorado por .gitignore)
├── notebooks/
│   ├── 03_evaluation.ipynb   # Notebook para evaluación detallada del modelo
│   └── eda.ipynb             # Notebook para Análisis Exploratorio de Datos
├── src/
│   ├── __init__.py           # Hace que src sea un paquete de Python
│   ├── api.py                # Script para la API con FastAPI
│   ├── config.py             # Archivo de configuración
│   ├── data_prep.py          # Funciones de preprocesamiento de datos
│   ├── generate_data.py      # Script para generar el dataset sintético
│   ├── predict.py            # Script para hacer predicciones
│   ├── train.py              # Script para entrenar el modelo
│   └── utils.py              # Funciones de utilidad (e.g., guardar metadata)
├── .gitignore
├── requirements.txt
└── README.md
```

## 🛠️ Instalación y Setup

Para configurar el entorno y ejecutar el proyecto, sigue estos pasos:

1.  **Clona el repositorio:**
    ```bash
    git clone <URL-del-repositorio>
    cd churn-classification
    ```

2.  **Crea un entorno virtual (recomendado):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # En Windows: venv\Scripts\activate
    ```

3.  **Instala las dependencias:**
    ```bash
    pip install -r requirements.txt
    ```

## ⚙️ Ejecución del Pipeline

Puedes ejecutar los diferentes componentes del pipeline desde la raíz del proyecto.

1.  **Generar los datos (si no existen):**
    El dataset `churn_data.csv` ya está incluido, pero si necesitas regenerarlo, puedes usar:
    ```bash
    python -m src.generate_data
    ```

2.  **Entrenar el modelo:**
    Este comando ejecutará el pipeline completo: cargará los datos, los preprocesará, entrenará un modelo de Regresión Logística, y guardará el modelo entrenado y sus métricas en la carpeta `models/`.
    ```bash
    python -m src.train
    ```

3.  **Hacer una predicción:**
    Puedes usar el script de predicción para obtener un resultado para un cliente de ejemplo.
    ```bash
    python -m src.predict
    ```

## 🌐 API de Predicción

El proyecto incluye una API creada con FastAPI para servir el modelo.

1.  **Asegúrate de que el modelo exista:**
    Si no lo has hecho, ejecuta el script de entrenamiento (`python -m src.train`) para generar el archivo `models/churn_model.pkl`.

2.  **Inicia el servidor de la API:**
    ```bash
    uvicorn src.api:app --host 0.0.0.0 --port 8000
    ```

3.  **Realiza una predicción vía API:**
    Puedes usar `curl` o cualquier cliente de API para enviar una solicitud POST al endpoint `/predict`.

    Ejemplo con `curl`:
    ```bash
    curl -X 'POST' \
      'http://127.0.0.1:8000/predict' \
      -H 'accept: application/json' \
      -H 'Content-Type: application/json' \
      -d '{
        "age": 42,
        "tenure": 8,
        "monthly_charges": 85,
        "contract_type": "Month-to-month",
        "internet_service": "Fiber optic"
      }'
    ```

    La API devolverá una predicción en formato JSON:
    ```json
    {
      "churn_prediction": 0
    }
    ```
