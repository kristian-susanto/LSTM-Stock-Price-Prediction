import os
import json
import time
import bcrypt
import streamlit as st
from datetime import datetime
from utils.db import get_users_collection, get_registration_access_collection, get_delete_requests_collection, get_password_reset_requests_collection
from dotenv import load_dotenv

LOCKOUT_THRESHOLD = 5
LOCKOUT_TIME = 300

load_dotenv()

def hash_password(password):
    """Mengubah kata sandi menjadi hash menggunakan bcrypt."""
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def register_user(username, password, admin_code=None):
    """Mendaftarkan pengguna baru ke database."""
    from dotenv import load_dotenv
    load_dotenv()
    ADMIN_REGISTRATION_CODE = os.getenv("ADMIN_REGISTRATION_CODE")

    users_col = get_users_collection()

    if users_col.find_one({"username": username}):
        return False

    role = "admin" if admin_code and admin_code == ADMIN_REGISTRATION_CODE else "user"

    user = {
        "username": username,
        "password": hash_password(password),
        "attempts": 0,
        "lockout_until": 0,
        "role": role,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    users_col.insert_one(user)
    return True

def verify_password(password, hashed):
    """Memverifikasi kecocokan kata sandi dengan hash."""
    return bcrypt.checkpw(password.encode(), hashed.encode())

def authenticate(username, password):
    """Mengautentikasi pengguna dan menangani lockout setelah terlalu banyak percobaan gagal."""
    users_col = get_users_collection()
    user = users_col.find_one({"username": username})

    if not user:
        return False, None, None

    now = time.time()

    if user.get("lockout_until", 0) > now:
        return False, 0, user["lockout_until"]
    elif user.get("lockout_until", 0) < now and user.get("attempts", 0) >= LOCKOUT_THRESHOLD:
        users_col.update_one({"username": username}, {
            "$set": {"attempts": 0, "lockout_until": 0}
        })
        user = users_col.find_one({"username": username})


    if verify_password(password, user["password"]):
        users_col.update_one({"username": username}, {
            "$set": {"attempts": 0}
        })
        return user.get("role", "user"), None, None
    else:
        attempts = user.get("attempts", 0) + 1
        lockout_until = time.time() + LOCKOUT_TIME if attempts >= LOCKOUT_THRESHOLD else 0

        users_col.update_one({"username": username}, {
            "$set": {
                "attempts": attempts,
                "lockout_until": lockout_until
            }
        })

        attempts_left = max(0, LOCKOUT_THRESHOLD - attempts)
        return False, attempts_left, lockout_until

def get_user_role(username):
    """Mengambil peran (role) dari pengguna berdasarkan username."""
    users_col = get_users_collection()
    user = users_col.find_one({"username": username})
    return user.get("role", "user") if user else "user"

def reset_password(username, new_password, old_password=None):
    """Mereset kata sandi pengguna dengan validasi opsional terhadap kata sandi lama."""
    users_col = get_users_collection()
    user = users_col.find_one({"username": username})

    if not user:
        return False, "Nama akun tidak ditemukan."

    if old_password:
        if not verify_password(old_password, user["password"]):
            return False, "Kata sandi lama salah."
    
    hashed = hash_password(new_password)
    users_col.update_one({"username": username}, {"$set": {"password": hashed, "attempts": 0, "lockout_until": 0}})
    return True, "Kata sandi berhasil di-reset."

def delete_user(username):
    """Menghapus akun pengguna dari database."""
    users_col = get_users_collection()
    result = users_col.delete_one({"username": username})
    return result.deleted_count > 0

def request_password_reset(username, new_password_hashed):
    """Mengirim permintaan reset kata sandi untuk diproses admin."""
    reset_requests_col = get_password_reset_requests_collection()
    users_col = get_users_collection()

    if not users_col.find_one({"username": username}):
        return False, "Nama akun tidak ditemukan."

    if reset_requests_col.find_one({"username": username, "status": "pending"}):
        return False, "Permintaan reset kata sandi sudah ada untuk akun ini."

    request = {
        "username": username,
        "new_password_hashed": new_password_hashed,
        "request_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending"
    }
    reset_requests_col.insert_one(request)
    return True, "Permintaan reset kata sandi telah dikirim ke admin."

def get_pending_password_reset_requests():
    """Mengambil daftar permintaan reset kata sandi yang belum diselesaikan."""
    reset_requests_col = get_password_reset_requests_collection()
    return list(reset_requests_col.find({"status": "pending"}))

def resolve_password_reset_request(request_id, action="reset"):
    """Menyetujui atau menolak permintaan reset kata sandi berdasarkan ID permintaan."""
    reset_requests_col = get_password_reset_requests_collection()
    users_col = get_users_collection()

    request = reset_requests_col.find_one({"_id": request_id})
    if request:
        username = request["username"]
        
        if action == "reset":
            new_password_hashed = request["new_password_hashed"]
            result = users_col.update_one({"username": username}, {
                "$set": {"password": new_password_hashed, "attempts": 0, "lockout_until": 0}
            })
            if result.modified_count > 0:
                reset_requests_col.update_one({"_id": request_id}, {"$set": {"status": "completed", "resolution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}})
                return True, f"Kata sandi untuk '{username}' berhasil di-reset dan permintaan diselesaikan."
            else:
                reset_requests_col.update_one({"_id": request_id}, {"$set": {"status": "failed_user_not_found", "resolution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}})
                return False, f"Gagal mereset kata sandi untuk '{username}'. Akun mungkin sudah tidak ada."
        elif action == "rejected":
            reset_requests_col.update_one({"_id": request_id}, {"$set": {"status": "rejected", "resolution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}})
            return True, f"Permintaan reset kata sandi untuk '{username}' berhasil ditolak."
    return False, "Permintaan tidak ditemukan."

def request_delete_user(username):
    """Mengirim permintaan penghapusan akun kepada admin."""
    delete_requests_col = get_delete_requests_collection()
    if delete_requests_col.find_one({"username": username, "status": "pending"}):
        return False, "Permintaan penghapusan akun sudah ada."
    
    users_col = get_users_collection()
    if not users_col.find_one({"username": username}):
        return False, "Nama akun tidak ditemukan."

    request = {
        "username": username,
        "request_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "pending"
    }
    delete_requests_col.insert_one(request)
    return True, "Permintaan penghapusan akun telah dikirim ke admin."

def get_pending_delete_requests():
    """Mengambil daftar permintaan penghapusan akun yang belum diselesaikan."""
    delete_requests_col = get_delete_requests_collection()
    return list(delete_requests_col.find({"status": "pending"}))

def resolve_delete_request(request_id, action="deleted"):
    """Menyetujui atau menolak permintaan penghapusan akun berdasarkan ID permintaan."""
    delete_requests_col = get_delete_requests_collection()
    users_col = get_users_collection()
    
    request = delete_requests_col.find_one({"_id": request_id})
    if request:
        if action == "deleted":
            if users_col.delete_one({"username": request["username"]}).deleted_count > 0:
                delete_requests_col.update_one({"_id": request_id}, {"$set": {"status": "completed", "resolution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}})
                return True, f"Akun '{request['username']}' berhasil dihapus dan permintaan diselesaikan."
            else:
                delete_requests_col.update_one({"_id": request_id}, {"$set": {"status": "failed_user_not_found", "resolution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}})
                return False, f"Gagal menghapus akun '{request['username']}'. Akun mungkin sudah tidak ada."
        elif action == "rejected":
            delete_requests_col.update_one({"_id": request_id}, {"$set": {"status": "rejected", "resolution_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}})
            return True, f"Permintaan penghapusan akun untuk '{request['username']}' berhasil ditolak."
    return False, "Permintaan tidak ditemukan."

def is_registration_enabled():
    """Memeriksa apakah fitur pendaftaran pengguna baru diaktifkan."""
    meta_col = get_registration_access_collection()
    meta = meta_col.find_one({"_id": "settings"}) or {}
    return meta.get("register_enabled", True)

def toggle_registration(enable: bool):
    """Mengaktifkan atau menonaktifkan fitur pendaftaran pengguna baru."""
    meta_col = get_registration_access_collection()
    meta_col.update_one({"_id": "settings"}, {
        "$set": {"register_enabled": enable}
    }, upsert=True)

def show_auth_sidebar():
    """Menampilkan informasi autentikasi di sidebar dan menangani proses logout."""
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    
    if st.session_state.get("logged_in", False):
        st.sidebar.header("Autentikasi")
        role = st.session_state.get("role", "user")
        st.sidebar.success(f"Nama akun: {st.session_state.username} ({role})")

        if st.sidebar.button("Keluar"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.success("Keluar berhasil!")

            st.markdown("""
                <meta http-equiv="refresh" content="3; url=/">
            """, unsafe_allow_html=True)

            st.stop()

        return role
    else:
        return None

def show_auth_page():
    """Menampilkan halaman autentikasi jika pengguna belum login."""
    if "logged_in" in st.session_state and st.session_state.logged_in:
        st.warning("Anda sudah masuk.")
        st.stop()