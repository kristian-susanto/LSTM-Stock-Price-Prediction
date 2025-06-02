import os
import json
import bcrypt
import time

USERS_FILE = "datas/users/users.json"
META_FILE = "datas/users/meta.json"
LOCKOUT_THRESHOLD = 5
LOCKOUT_TIME = 300

def init_files():
    os.makedirs("datas/users", exist_ok=True)
    if not os.path.exists(USERS_FILE):
        with open(USERS_FILE, "w") as f:
            json.dump({}, f)
    if not os.path.exists(META_FILE):
        with open(META_FILE, "w") as f:
            json.dump({"register_enabled": True}, f)
            
def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    try:
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

def verify_password(password, hashed):
    return bcrypt.checkpw(password.encode(), hashed.encode())

def authenticate(username, password):
    users = load_users()
    user = users.get(username)
    if not user:
        return False

    if user.get("lockout_until", 0) > time.time():
        return False

    if verify_password(password, user["password"]):
        user["attempts"] = 0
        save_users(users)
        return True
    else:
        user["attempts"] = user.get("attempts", 0) + 1
        if user["attempts"] >= LOCKOUT_THRESHOLD:
            user["lockout_until"] = time.time() + LOCKOUT_TIME
        save_users(users)
        return False

def register_user(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = {
        "password": hash_password(password),
        "attempts": 0,
        "lockout_until": 0
    }
    save_users(users)
    return True

def is_registration_enabled():
    if not os.path.exists(META_FILE):
        return True
    with open(META_FILE, "r") as f:
        meta = json.load(f)
    return meta.get("register_enabled", True)

def toggle_registration(enable: bool):
    meta = {"register_enabled": enable}
    with open(META_FILE, "w") as f:
        json.dump(meta, f, indent=2)

def delete_user(username):
    users = load_users()
    if username in users:
        del users[username]
        save_users(users)
        return True
    return False
