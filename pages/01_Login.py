import time
import streamlit as st
from utils.auth import authenticate

st.set_page_config(
    page_title="Masuk",
    page_icon="assets/favicon.ico",
    layout="wide"
)

st.header("Login", divider="gray")

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

username = st.text_input("Nama Akun")
password = st.text_input("Kata Sandi", type="password")

if st.button("Masuk"):
    if authenticate(username, password):
        st.session_state.logged_in = True
        st.session_state.username = username
        
        countdown = st.empty()
        for i in range(5, 0, -1):
            countdown.success("Masuk berhasil! Silahkan menuju ke menu admin panel yang ada di sidebar.")
            time.sleep(1)

        st.query_params.clear()
        st.rerun()
    else:
        st.error("Nama akun atau kata sandi salah.")
