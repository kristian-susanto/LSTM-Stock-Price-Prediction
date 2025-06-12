import re
import streamlit as st
from utils.auth import is_registration_enabled, show_auth_sidebar, show_auth_page, register_user

# Konfigurasi halaman Streamlit
st.set_page_config(page_title="Registrasi", page_icon="assets/favicon.ico", layout="wide")
st.header("Register", divider="gray")

# Tampilkan sidebar login jika sudah login
role = show_auth_sidebar()
is_admin = role == "admin" if role else False

# Mencegah akses halaman jika sudah login
show_auth_page()

# Cek apakah registrasi masih diizinkan
if not is_registration_enabled():
    st.warning("Maaf, Anda tidak dapat melakukan registrasi.")
    st.stop()

# Penjelasan fungsi halaman registrasi
st.markdown(
    f"""
    <div style='text-align: justify; margin-bottom: 10px'>
        Halaman ini digunakan oleh pengguna baru yang ingin membuat akun.
    </div>
    """,
    unsafe_allow_html=True
)

# Form input registrasi
username = st.text_input("Nama Akun")
password = st.text_input("Kata Sandi", type="password")
confirm = st.text_input("Konfirmasi Kata Sandi", type="password")

# Aturan validasi password: panjang 3-20 karakter, hanya huruf, angka, atau underscore
password_pattern = r"^[a-zA-Z0-9_]{3,20}$"

# Input opsional untuk kode admin jika akan mendaftar sebagai admin
admin_code = st.text_input("Kode Admin (Opsional)", type="password")

# Tombol daftar ditekan
if st.button("Daftar"):
    # Validasi konfirmasi kata sandi
    if password != confirm:
        st.error("Kata sandi tidak cocok.")
    # Validasi format kata sandi
    elif not re.match(password_pattern, password):
        st.error("Kata sandi harus terdiri dari 3 hingga 20 karakter, dan hanya boleh mengandung huruf (a-z atau A-Z), angka (0-9), dan garis bawah (_). Spasi dan karakter khusus lainnya tidak diperbolehkan.")
    # Proses pendaftaran jika valid
    elif register_user(username, password, admin_code=admin_code):
        st.success("Pendaftaran berhasil! Mengarahkan ke halaman login...")
        # Redirect otomatis ke halaman login setelah 5 detik
        st.markdown("""
            <meta http-equiv="refresh" content="5; url=/Login">
        """, unsafe_allow_html=True)
        st.stop()
    # Jika username sudah digunakan
    else:
        st.error("Nama akun sudah ada.")