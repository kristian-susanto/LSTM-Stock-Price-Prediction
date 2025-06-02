import re
import streamlit as st
from utils.auth import init_files, register_user, is_registration_enabled

st.set_page_config(
    page_title="Registrasi",
    page_icon="assets/favicon.ico",
    layout="wide"
)

init_files()

st.header("Register", divider="gray")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if st.session_state.logged_in:
    st.sidebar.header("Autentikasi")
    st.sidebar.success(f"Nama akun: {st.session_state.username}")
    is_admin = True
    if st.sidebar.button("Keluar"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Keluar berhasil!")

        st.markdown("""
            <meta http-equiv="refresh" content="3; url=/">
        """, unsafe_allow_html=True)

        st.stop()
else:
    is_admin = False

if not is_registration_enabled():
    st.warning("Fitur register dinonaktifkan oleh admin.")
    st.stop()

username = st.text_input("Nama Akun")
password = st.text_input("Kata Sandi", type="password")
confirm = st.text_input("Konfirmasi Kata Sandi", type="password")

password_pattern = r"^[a-zA-Z0-9_]{3,20}$"

if st.button("Daftar"):
    if password != confirm:
        st.error("Kata sandi tidak cocok.")
    elif not re.match(password_pattern, password):
        st.error("Kata sandi harus 3-20 karakter dan hanya boleh terdiri dari huruf, angka, dan underscore (_).")
    elif register_user(username, password):
        st.success("Pendaftaran berhasil! Mengarahkan ke halaman login...")
        st.markdown("""
            <meta http-equiv="refresh" content="1; url=/Login">
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.error("Nama akun sudah ada.")
