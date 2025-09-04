import json
from datetime import datetime

def save_metadata(metrics: dict, path="models/metadata.json"):
    metadata = {
        "created_at": datetime.utcnow().isoformat(),
        "metrics": metrics
    }
    with open(path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata guardada en {path}")
