import io
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from utils.auth import load_users, save_users, hash_password, is_registration_enabled, toggle_registration, delete_user
from utils.model_utils import init_model_dirs
from utils.model_utils import list_models, load_model_file, get_model_info, delete_model
from utils.model_utils import list_histories, load_history_file, get_history_info, delete_history
from utils.model_utils import list_metadatas, load_param_file, get_param_info, delete_metadata

st.set_page_config(
    page_title="Panel Admin",
    page_icon="assets/favicon.ico",
    layout="wide"
)

init_model_dirs() 

st.header("Admin Panel", divider="gray")

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "selected_history" not in st.session_state:
    st.session_state.selected_history = None
if "selected_metadata" not in st.session_state:
    st.session_state.selected_metadata = None

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

if not st.session_state.get("logged_in", False):
    st.warning("Hanya admin yang dapat mengakses halaman ini.")
    st.stop()

if st.session_state.username == "admin":
    with st.sidebar:
        st.subheader("Kontrol Registrasi Admin")
        reg_enabled = is_registration_enabled()
        if st.button("Aktifkan Registrasi" if not reg_enabled else "Nonaktifkan Registrasi"):
            toggle_registration(not reg_enabled)
            st.rerun()

        st.subheader("Tambah Akun Baru")
        with st.form("form_tambah_user"):
            new_username = st.text_input("Nama Akun Baru")
            submit_add_user = st.form_submit_button("Tambah")

            if submit_add_user:
                users = load_users()
                if new_username in users:
                    st.error("Username sudah terdaftar.")
                elif new_username.strip() == "":
                    st.warning("Username tidak boleh kosong.")
                else:
                    default_password = "11111111"
                    users[new_username] = hash_password(default_password)
                    save_users(users)
                    st.success(f"Pengguna '{new_username}' berhasil ditambahkan dengan password default.")
                    st.rerun()

st.subheader("Manajemen Model")
models = list_models()

if models:
    model_data = []
    for model_name in models:
        info = get_model_info(model_name)
        model_data.append({
            "Nama Model": model_name,
            "Tanggal": info.get("created_at", "Tidak diketahui")
        })

    df_models = pd.DataFrame(model_data)

    if st.session_state.username == "admin":
        st.dataframe(df_models[["Tanggal", "Nama Model"]], use_container_width=True)
    else:
        st.dataframe(df_models[["Nama Model"]], use_container_width=True)

    selected_model = st.selectbox("Pilih Model", models)
    if st.button("Buka Detail Model Terpilih", key="buka_model"):
        st.session_state.selected_model = selected_model
        st.rerun()

    if st.session_state.username == "admin":
        if st.button("Hapus Model Terpilih", key="hapus_model"):
            delete_model(selected_model)
            st.success(f"Model '{selected_model}' berhasil dihapus.")
            st.rerun()
else:
    st.info("Belum ada model yang disimpan.")

if st.session_state.selected_model:
    st.markdown(f"#### Detail Model: {st.session_state.selected_model}")
    try:
        model = load_model_file(st.session_state.selected_model)
        buffer = io.StringIO()
        model.summary(print_fn=lambda x: buffer.write(x + "\n"))
        st.code(buffer.getvalue())

        if st.button("Tutup Detail", key="tutup_detail_model"):
            st.session_state.selected_model = None
            st.rerun()
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        if st.button("Tutup Detail", key="tutup_detail_model"):
            st.session_state.selected_model = None
            st.rerun()

st.divider()
st.subheader("Manajemen Histori Training")
histories = list_histories()

if histories:
    history_data = []
    for history_file in histories:
        info = get_history_info(history_file)
        history_data.append({
            "Nama Histori": history_file,
            "Tanggal": info.get("created_at", "Tidak diketahui")
        })

    df_histories = pd.DataFrame(history_data)

    if st.session_state.username == "admin":
        st.dataframe(df_histories[["Tanggal", "Nama Histori"]], use_container_width=True)
    else:
        st.dataframe(df_histories[["Nama Histori"]], use_container_width=True)

    selected_history = st.selectbox("Pilih Histori", histories)
    if st.button("Buka Detail Histori Terpilih", key="buka_histori"):
        st.session_state.selected_history = selected_history
        st.rerun()

    if st.session_state.username == "admin":
        if st.button("Hapus Histori Terpilih", key="hapus_histori"):
            delete_history(selected_history)
            st.success(f"Histori '{selected_history}' berhasil dihapus.")
            st.rerun()
else:
    st.info("Belum ada histori training yang disimpan.")

if st.session_state.selected_history:
    st.markdown(f"#### Detail Histori: `{st.session_state.selected_history}`")

    history_dict = load_history_file(st.session_state.selected_history)

    if history_dict:
        st.json(history_dict)

        if "loss" in history_dict:
            st.markdown("#### Visualisasi Training Loss")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(history_dict["loss"], label="Training Loss", color="blue")
            if "val_loss" in history_dict:
                ax.plot(history_dict["val_loss"], label="Validation Loss", color="orange")
            ax.set_title("Training Loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        else:
            st.warning("Data 'loss' tidak ditemukan dalam file JSON.")

        if st.button("Tutup Detail", key="tutup_detail_histori"):
            st.session_state.selected_history = None
            st.rerun()
    else:
        st.warning("Gagal memuat isi file histori.")

st.divider()
st.subheader("Manajemen Metadata Parameter dan Evaluasi")
metadatas = list_metadatas()

if metadatas:
    metadata_data = []
    for metadata_file in metadatas:
        info = get_param_info(metadata_file)
        metadata_data.append({
            "Nama Metadata": metadata_file,
            "Tanggal": info.get("created_at", "Tidak diketahui")
        })

    df_metadatas = pd.DataFrame(metadata_data)

    if st.session_state.username == "admin":
        st.dataframe(df_metadatas[["Tanggal", "Nama Metadata"]], use_container_width=True)
    else:
        st.dataframe(df_metadatas[["Nama Metadata"]], use_container_width=True)

    selected_metadata = st.selectbox("Pilih Metadata Parameter dan Evaluasi", metadatas)
    if st.button("Buka Detail Metadata Terpilih", key="buka_metadata"):
        st.session_state.selected_metadata = selected_metadata
        st.rerun()

    if st.session_state.username == "admin":
        if st.button("Hapus Metadata Terpilih", key="hapus_metadata"):
            delete_metadata(selected_metadata)
            st.success(f"Metadata '{selected_metadata}' berhasil dihapus.")
            st.rerun()
else:
    st.info("Belum ada metadata parameter dan evaluasi yang disimpan.")

if st.session_state.selected_metadata:
    st.markdown(f"#### Detail Metadata Parameter dan Evaluasi: `{st.session_state.selected_metadata}`")

    metadata_dict = load_param_file(st.session_state.selected_metadata)

    if metadata_dict:
        st.json(metadata_dict)

        if st.button("Tutup Detail", key="tutup_detail_metadata"):
            st.session_state.selected_metadata = None
            st.rerun()
    else:
        st.warning("Gagal memuat isi file metadata parameter dan evaluasi.")

st.divider()
st.subheader("Manajemen Pengguna")

users = load_users()
usernames = sorted(users.keys())

if not st.session_state.username == "admin":
    usernames = [u for u in usernames if u != "admin"]

if usernames:
    user_data = []
    for username in usernames:
        user_data.append({
            "Nama Pengguna": username
        })

    df_users = pd.DataFrame(user_data)

    st.dataframe(df_users, use_container_width=True)

    selected_user = st.selectbox("Pilih Pengguna", usernames)

    if st.session_state.username == "admin":
        if selected_user != "admin":
            if st.button("Hapus Pengguna", key="hapus_pengguna"):
                delete_user(selected_user)
                st.success(f"Pengguna '{selected_user}' dihapus.")
                st.rerun()
            if st.button("Reset Password", key="reset_password"):
                default_password = "11111111"
                users[selected_user] = hash_password(default_password)
                save_users(users)
                st.success(f"Password untuk '{selected_user}' telah direset ke default.")
                st.rerun()
        else:
            st.info("Akun admin tidak dapat dihapus atau direset.")
else:
    st.info("Belum ada pengguna yang terdaftar.")
