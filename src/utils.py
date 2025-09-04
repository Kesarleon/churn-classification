import json
import logging
from datetime import datetime

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_metadata(metrics: dict, path: str):
    """Guarda un diccionario de metadatos en un archivo JSON."""
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    logging.info(f"Metadata guardada en {path}")
