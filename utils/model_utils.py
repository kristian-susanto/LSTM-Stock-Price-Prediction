import os
import json
from tensorflow.keras.models import load_model
from datetime import datetime

MODEL_DIR = "datas/models"
HISTORY_DIR = "datas/histories"
METADATA_DIR = "datas/metadatas"

def init_model_dirs():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)
    os.makedirs(METADATA_DIR, exist_ok=True)

def list_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    
def save_model(model, filename):
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    full_path = os.path.join(MODEL_DIR, filename)
    model.save(full_path)

    meta_path = full_path.replace(".keras", ".json")
    metadata = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
def load_model_file(filename):
    return load_model(os.path.join(MODEL_DIR, filename))

def get_model_info(filename):
    meta_path = os.path.join(MODEL_DIR, filename.replace(".keras", ".json"))
    model_path = os.path.join(MODEL_DIR, filename)

    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            return json.load(f)
    
    if os.path.exists(model_path):
        created_timestamp = os.path.getctime(model_path)
        created_at = datetime.fromtimestamp(created_timestamp).strftime("%Y-%m-%d %H:%M:%S")
        return {"created_at": created_at}

    return {"created_at": "Tidak diketahui"}

def delete_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    meta_path = path.replace(".keras", ".json")
    
    if os.path.exists(path):
        os.remove(path)
    if os.path.exists(meta_path):
        os.remove(meta_path)

def list_histories():
    return [f for f in os.listdir(HISTORY_DIR) if f.endswith(".json")]

def save_training_history(history, filename):
    if not os.path.exists(HISTORY_DIR):
        os.makedirs(HISTORY_DIR)
    history_data = history.history if hasattr(history, "history") else history
    history_path = os.path.join(HISTORY_DIR, filename.replace(".keras", "_history.json"))
    with open(history_path, "w") as f:
        json.dump(history_data, f, indent=2)

def load_history_file(history_filename):
    history_path = os.path.join(HISTORY_DIR, history_filename)
    if os.path.exists(history_path):
        with open(history_path, "r") as f:
            history = json.load(f)
        return history
    return None

def get_history_info(filename):
    path = os.path.join(HISTORY_DIR, filename)
    if os.path.exists(path):
        created_time = os.path.getctime(path)
        created_at = datetime.fromtimestamp(created_time).strftime("%Y-%m-%d %H:%M:%S")
        return {"created_at": created_at}
    return {"created_at": "Tidak diketahui"}

def delete_history(filename):
    path = os.path.join(HISTORY_DIR, filename)
    if os.path.exists(path):
        os.remove(path)

def list_metadatas():
    return [f for f in os.listdir(METADATA_DIR) if f.endswith(".json")]

def save_training_metadata(metadata, filename):
    if not os.path.exists(METADATA_DIR):
        os.makedirs(METADATA_DIR)
    metadata_data = metadata.metadata if hasattr(metadata, "metadata") else metadata
    metadata_path = os.path.join(METADATA_DIR, filename.replace(".keras", "_params.json"))
    with open(metadata_path, "w") as f:
        json.dump(metadata_data, f, indent=2)

def load_param_file(metadata_filename):
    metadata_path = os.path.join(METADATA_DIR, metadata_filename)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        return metadata
    return None

def get_param_info(filename):
    path = os.path.join(METADATA_DIR, filename)
    if os.path.exists(path):
        created_time = os.path.getctime(path)
        created_at = datetime.fromtimestamp(created_time).strftime("%Y-%m-%d %H:%M:%S")
        return {"created_at": created_at}
    return {"created_at": "Tidak diketahui"}

def delete_metadata(filename):
    path = os.path.join(METADATA_DIR, filename)
    if os.path.exists(path):
        os.remove(path)