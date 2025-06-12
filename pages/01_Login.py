import time
import streamlit as st
from utils.auth import hash_password, verify_password, authenticate, LOCKOUT_THRESHOLD, get_users_collection, reset_password, request_password_reset, delete_user, request_delete_user, show_auth_sidebar, show_auth_page

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Masuk", page_icon="assets/favicon.ico", layout="wide")
st.header("Login", divider="gray")

# Tampilkan info login di sidebar jika sudah login
role = show_auth_sidebar()
is_admin = role == "admin" if role else False

# Mencegah pengguna membuka halaman login jika sudah login
show_auth_page()

# Instruksi untuk pengguna
st.markdown(
    f"""
    <div style='text-align: justify; margin-bottom: 10px'>
        Pengguna dapat masuk dengan mengisi nama akun dan kata sandi.
    </div>
    """,
    unsafe_allow_html=True
)

# Form input login
username = st.text_input("Nama Akun")
password = st.text_input("Kata Sandi", type="password")

# Tombol masuk
if st.button("Masuk"):
    # Autentikasi pengguna
    role, attempts_left, lockout_until = authenticate(username, password)
    
    if role:
        # Simpan session state dan redirect ke dashboard jika berhasil
        st.session_state.logged_in = True
        st.session_state.username = username
        st.session_state.role = role
        
        countdown = st.empty()
        for i in range(5, 0, -1):
            countdown.success("Masuk berhasil! Silakan menuju ke menu dashboard di sidebar.")
            time.sleep(1)

        st.query_params.clear()
        st.rerun()
    else:
        # Tampilkan pesan error atau lockout jika gagal
        if attempts_left == 0 and lockout_until:
            unlock_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(lockout_until))
            st.error(f"Akun terkunci. Harap tunggu hingga {unlock_time} untuk mencoba kembali.")
        elif attempts_left is not None:
            st.warning(f"Nama akun atau kata sandi salah. Kesempatan tersisa: {LOCKOUT_THRESHOLD - attempts_left} dari {LOCKOUT_THRESHOLD}.") # Corrected attempts_left display
        else:
            st.error("Nama akun atau kata sandi salah.")

st.divider()

# Reset Kata Sandi
st.subheader("Reset Kata Sandi")
st.markdown(
    f"""
    <div style='text-align: justify; margin-bottom: 10px'>
        Pengguna dapat mengganti kata sandi, maka dapat memilih salah satu cara seperti berikut.
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Reset langsung jika tahu kata sandi lama
st.markdown("#### 1. Reset Kata Sandi Langsung (Tahu Kata Sandi Lama)")
st.write("Masukkan nama akun, kata sandi lama, dan kata sandi baru Anda.")
username_reset_known = st.text_input("Nama Akun", key="reset_user_known")
old_password_known = st.text_input("Kata Sandi Lama", type="password", key="old_pass_known")
new_password_known = st.text_input("Kata Sandi Baru", type="password", key="new_pass_known")
confirm_reset = st.checkbox("Saya yakin ingin mereset kata sandi ini.", key="confirm_reset")

# Tombol reset password langsung
if st.button("Reset Kata Sandi"):
    if username_reset_known and old_password_known and new_password_known and confirm_reset:
        success, message = reset_password(username_reset_known, new_password_known, old_password_known)
        if success:
            st.success(message)
        else:
            st.error(message)
    else:
        st.warning("Mohon lengkapi semua kolom untuk reset kata sandi.")

st.divider()

# 2. Ajukan permintaan reset ke admin
st.markdown("#### 2. Ajukan Permintaan Reset Kata Sandi (Lupa Kata Sandi Lama)")
st.write("Masukkan nama akun dan kata sandi baru yang diinginkan jika Anda lupa kata sandi lama. Permintaan ini akan dikirimkan kepada admin untuk persetujuan.")
username_request_reset = st.text_input("Nama Akun untuk Ajukan Reset", key="request_reset_user")
new_password_request = st.text_input("Kata Sandi Baru yang Diinginkan", type="password", key="request_new_pass")
confirm_reset_request = st.checkbox("Saya yakin ingin mengajukan permintaan reset kata sandi ini.", key="confirm_reset_request")

# Tombol ajukan permintaan reset
if st.button("Ajukan Permintaan Reset", key="btn_request_reset") and confirm_reset_request:
    if username_request_reset and new_password_request:
        hashed_new_password = hash_password(new_password_request)
        success, message = request_password_reset(username_request_reset, hashed_new_password)
        if success:
            st.success(message)
        else:
            st.warning(message)
    else:
        st.warning("Mohon lengkapi nama akun dan kata sandi baru yang diinginkan.")

st.divider()

# Hapus Akun
st.subheader("Hapus Akun")
st.markdown(
    f"""
    <div style='text-align: justify; margin-bottom: 10px'>
        Jika pengguna ingin menghapus akun, maka dapat memilih salah satu cara seperti berikut.
    </div>
    """,
    unsafe_allow_html=True
)

# 1. Hapus akun jika tahu kata sandi
st.markdown("#### 1. Hapus Akun Langsung (Tahu Kata Sandi)")
st.write("Masukkan nama akun dan kata sandi untuk menghapus akun. **Langkah ini permanen dan tidak bisa dibatalkan.**")
username_delete = st.text_input("Nama Akun untuk Dihapus", key="delete_user")
password_delete = st.text_input("Kata Sandi", type="password", key="delete_pass")
confirm_delete = st.checkbox("Saya yakin ingin menghapus akun ini.", key="confirm_delete_with_pass")

# Tombol hapus akun langsung
if st.button("Hapus Akun", key="btn_delete_with_pass") and confirm_delete:
    user_col = get_users_collection()
    user = user_col.find_one({"username": username_delete})

    if user:
        if verify_password(password_delete, user["password"]):
            if delete_user(username_delete):
                st.success("Akun berhasil dihapus.")
                # Logout otomatis jika akun sendiri dihapus
                if st.session_state.get("username") == username_delete:
                    st.session_state.logged_in = False
                    st.session_state.username = ""
                    st.rerun()
            else:
                st.error("Terjadi kesalahan saat menghapus akun.")
        else:
            st.error("Kata sandi salah.")
    else:
        st.error("Nama akun tidak ditemukan.")

st.divider()

# 2. Ajukan permintaan penghapusan akun jika lupa kata sandi
st.markdown("#### 2. Ajukan Permintaan Penghapusan Akun Langsung (Lupa Kata Sandi)")
st.write("Masukkan nama akun jika Anda lupa kata sandi. Permintaan ini akan dikirimkan kepada admin untuk persetujuan.")
username_request_delete = st.text_input("Nama Akun yang ingin dihapus (lupa password)", key="request_delete_user")
confirm_request_delete = st.checkbox("Saya yakin ingin mengajukan permintaan penghapusan akun ini.", key="confirm_request_delete")

# Tombol ajukan permintaan hapus
if st.button("Ajukan Permintaan Penghapusan Akun", key="btn_request_delete") and confirm_request_delete:
    if username_request_delete:
        success, message = request_delete_user(username_request_delete)
        if success:
            st.success(message)
        else:
            st.warning(message)
    else:
        st.warning("Nama akun tidak boleh kosong untuk mengajukan permintaan.")