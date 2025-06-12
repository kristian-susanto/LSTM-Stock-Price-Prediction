import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]

def get_users_collection():
    """Mengambil koleksi pengguna dari database."""
    return db["users"]

def get_model_collection():
    """Mengambil koleksi model dari database."""
    return db["model"]

def get_model_metadata_collection():
    """Mengambil koleksi metadata model dari database."""
    return db["model_metadata"]

def get_registration_access_collection():
    """Mengambil koleksi pengaturan akses pendaftaran dari database."""
    return db["registration_access"]

def get_delete_requests_collection():
    """Mengambil koleksi permintaan penghapusan akun dari database."""
    return db["delete_requests"]

def get_password_reset_requests_collection():
    """Mengambil koleksi permintaan reset kata sandi dari database."""
    return db["password_reset_requests"]