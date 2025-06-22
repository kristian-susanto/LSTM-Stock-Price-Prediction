import io
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime
from utils.auth import hash_password, get_user_role, reset_password, get_pending_password_reset_requests, resolve_password_reset_request, delete_user, get_pending_delete_requests, resolve_delete_request, is_registration_enabled, toggle_registration, show_auth_sidebar
from utils.db import get_users_collection
from utils.model import list_model, load_model_file, get_model_file, delete_model_file, list_model_metadata, get_model_metadata_file, load_model_metadata_file, delete_model_metadata_file, extract_model_info

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Dashboard", page_icon="assets/favicon.ico", layout="wide")
st.header("Dashboard", divider="gray")

# Tampilkan sidebar informasi pengguna
role = show_auth_sidebar()
is_admin = role == "admin" if role else False

# Mengecek apakah pengguna sudah login jika belum maka hentikan akses
if not st.session_state.get("logged_in", False):
    st.warning("Masuk untuk yang dapat mengakses halaman ini.")
    st.stop()

# Menjabarkan state untuk pilihan model dan metadata model yang sedang dibuka
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "selected_model_metadata" not in st.session_state:
    st.session_state.selected_model_metadata = None

st.subheader("Manajemen Model")
st.markdown(
    f"""
    <div style='text-align: justify; margin-bottom: 10px'>
        Bagian ini menampilkan daftar model yang telah diunggah atau disimpan oleh pengguna. 
        Setiap model memiliki informasi seperti nama model, tanggal dibuat, pengguna yang mengunggah, 
        dan peran pengguna tersebut.
    </div>
    """,
    unsafe_allow_html=True
)

# Ambil daftar model yang tersedia
models = list_model()

if models:
    # Tampilkan model ke dalam DataFrame
    model_data = []
    for model_name in models:
        info = get_model_file(model_name)
        extracted_info = extract_model_info(model_name)
        model_data.append({
            "Tanggal Pembuatan": info.get("created_at", "Tidak diketahui"),
            "Nama Model": model_name,
            "Ticker Saham": extracted_info["ticker"],
            "Frekuensi": extracted_info["frekuensi"],
            "Tanggal Awal": extracted_info["tanggal_awal"],
            "Tanggal Akhir": extracted_info["tanggal_akhir"],
            "Tipe Model": extracted_info["tipe_model"],
            "Nama Akun": info.get("username", "-"),
            "Peran": info.get("role", "guest")
        })

    df_models = pd.DataFrame(model_data)

    # Melakukan pencarian ticker
    search_ticker_in_model = st.text_input("Cari Ticker Saham", "")

    # Filter DataFrame jika ada masukan pencarian
    if search_ticker_in_model:
        df_models = df_models[df_models["Ticker Saham"].str.contains(search_ticker_in_model, case=False)]
    
    if st.session_state.logged_in:
        st.dataframe(
            df_models[
                ["Tanggal Pembuatan", "Nama Model", "Ticker Saham", "Frekuensi", "Tanggal Awal", "Tanggal Akhir", "Tipe Model", "Nama Akun", "Peran"]
            ],
            use_container_width=True
        )
    else:
        st.dataframe(
            df_models[
                ["Tanggal Pembuatan", "Nama Model", "Ticker Saham", "Frekuensi", "Tanggal Awal", "Tanggal Akhir", "Tipe Model"]
            ],
            use_container_width=True
        )

    # Pilih model untuk dibuka atau dihapus
    selected_model = st.selectbox("Pilih Model", models)
    col1_detail, col2_delete = st.columns(2)
    # Tombol untuk membuka detail model
    with col1_detail:
        if st.button("Buka Detail Model Terpilih", key="buka_model"):
            st.session_state.selected_model = selected_model
            st.rerun()
    # Hanya admin yang bisa menghapus model
    with col2_delete:
        if get_user_role(st.session_state.username) == "admin":
            if st.button("Hapus Model Terpilih", key="hapus_model"):
                delete_model_file(selected_model)
                st.success(f"Model '{selected_model}' berhasil dihapus.")
                st.rerun()
else:
    st.info("Belum ada model yang disimpan.")

# Detail model terpilih
if st.session_state.selected_model:
    st.markdown(f"##### Detail Model: {st.session_state.selected_model}")
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

st.subheader("Manajemen Metadata Model")
st.markdown(
    f"""
    <div style='text-align: justify; margin-bottom: 10px'>
        Metadata adalah informasi tambahan yang disimpan bersamaan dengan model, 
        seperti riwayat pelatihan model (training history) dan parameter penting lainnya.
    </div>
    """,
    unsafe_allow_html=True
)

# Ambil semua metadata model
all_metadata = list_model_metadata()

if all_metadata:
    metadata_table_data = []
    for meta_item in all_metadata:
        info = get_model_metadata_file(meta_item["name"])
        extracted_info = extract_model_info(meta_item["name"])
        metadata_table_data.append({
            "Tanggal Pembuatan": info.get("created_at", "Tidak diketahui"),
            "Nama Metadata": meta_item["name"],
            "Ticker": extracted_info["ticker"],
            "Frekuensi": extracted_info["frekuensi"],
            "Tanggal Awal": extracted_info["tanggal_awal"],
            "Tanggal Akhir": extracted_info["tanggal_akhir"],
            "Tipe Model": extracted_info["tipe_model"],
            "Nama Akun": info.get("username", "-"),
            "Peran": info.get("role", "guest")
        })

    df_all_metadata = pd.DataFrame(metadata_table_data)

    # Melakukan pencarian ticker
    search_ticker_in_model_metadata = st.text_input("Cari Ticker", "")

    # Filter DataFrame jika ada masukan pencarian
    if search_ticker_in_model_metadata:
        df_all_metadata = df_all_metadata[df_all_metadata["Ticker"].str.contains(search_ticker_in_model_metadata, case=False)]

    if st.session_state.logged_in:
        st.dataframe(
            df_all_metadata[
                ["Tanggal Pembuatan", "Nama Metadata", "Ticker", "Frekuensi", "Tanggal Awal", "Tanggal Akhir", "Tipe Model", "Nama Akun", "Peran"]
            ],
            use_container_width=True
        )
    else:
        st.dataframe(
            df_all_metadata[
                ["Tanggal Pembuatan", "Nama Metadata", "Ticker", "Frekuensi", "Tanggal Awal", "Tanggal Akhir", "Tipe Model"]
            ],
            use_container_width=True
        )

    metadata_options = [item['name'] for item in all_metadata]
    selected_metadata_name = st.selectbox("Pilih Metadata Histori dan Parameter", metadata_options)
    
    # Tampilkan atau hapus metadata
    if selected_metadata_name:
        col1_detail, col2_delete = st.columns(2)
        with col1_detail:
            if st.button("Buka Detail Metadata Terpilih", key="buka_model_metadata"):
                st.session_state.selected_model_metadata = {"name": selected_metadata_name}
                st.rerun()
        with col2_delete:
            if get_user_role(st.session_state.username) == "admin":
                if st.button("Hapus Metadata Terpilih", key="hapus_model_metadata"):
                    delete_model_metadata_file(selected_metadata_name)
                    st.success(f"Metadata '{selected_metadata_name}' berhasil dihapus.")
                    st.session_state.selected_model_metadata = None
                    st.rerun()
else:
    st.info("Belum ada metadata histori training dan parameter yang disimpan.")

# Detail metadata model
if st.session_state.selected_model_metadata:
    selected_name = st.session_state.selected_model_metadata["name"]
    st.markdown(f"##### Detail Metadata Model: `{selected_name}`")

    combined_data_dict = load_model_metadata_file(selected_name)

    if combined_data_dict:
        st.json(combined_data_dict)

        # Visualisasi loss model jika tersedia
        if "history" in combined_data_dict and combined_data_dict["history"] and "loss" in combined_data_dict["history"]:
            st.markdown("#### Visualisasi Training Loss")
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(combined_data_dict["history"]["loss"], label="Training Loss", color="blue")
            if "val_loss" in combined_data_dict["history"]:
                ax.plot(combined_data_dict["history"]["val_loss"], label="Validation Loss", color="orange")
            ax.set_title("Training Loss")
            ax.set_xlabel("Epochs")
            ax.set_ylabel("Loss")
            ax.grid(True)
            ax.legend()
            st.pyplot(fig)
        else:
            st.info("Data 'loss' tidak ditemukan dalam histori yang disimpan.")

        if st.button("Tutup Detail", key="tutup_detail_model_metadata"):
            st.session_state.selected_model_metadata = None
            st.rerun()
    else:
        st.warning("Gagal memuat isi file data model.")

if get_user_role(st.session_state.username) == "admin":
    # Sidebar pada admin untuk kontrol registrasi + tambah akun
    with st.sidebar:
        st.subheader("Kontrol Registrasi Admin")
        reg_enabled = is_registration_enabled()
        if st.button("Aktifkan Registrasi" if not reg_enabled else "Nonaktifkan Registrasi"):
            toggle_registration(not reg_enabled)
            st.rerun()

        st.subheader("Tambah Akun Baru")
        with st.form("form_tambah_user"):
            new_username = st.text_input("Nama Akun Baru")
            selected_role = st.selectbox("Peran Akun", ["Pilih", "user", "admin"], key="selected_role")
            submit_add_user = st.form_submit_button("Tambah")

            # Validasi dan tambah akun
            if submit_add_user:
                users_col = get_users_collection()
                if users_col.find_one({"username": new_username}):
                    st.error("Nama akun sudah terdaftar.")
                elif new_username.strip() == "":
                    st.warning("Nama akun tidak boleh kosong.")
                elif selected_role.strip() == "Pilih":
                    st.warning("Peran akun tidak boleh kosong.")
                else:
                    default_password = "11111111"
                    user_doc = {
                        "username": new_username,
                        "password": hash_password(default_password),
                        "attempts": 0,
                        "lockout_until": 0,
                        "role": selected_role,
                        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    users_col.insert_one(user_doc)
                    st.success(f"Akun '{new_username}' berhasil ditambahkan dengan password default.")
                    st.rerun()

    st.divider()
    st.subheader("Manajemen Pengguna")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Bagian ini hanya dapat diakses oleh admin dan berfungsi untuk mengelola akun pengguna sistem
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("#### 1. Reset Kata Sandi, Menghapus Akun Langsung, dan Mengubah Peran Akun")

    # Tampilkan semua pengguna
    users_col = get_users_collection()
    users = {user["username"]: user for user in users_col.find()}
    usernames = sorted(users.keys())

    if usernames:
        user_data = []
        for username in usernames:
            user_info = users.get(username, {})
            user_data.append({
                "Nama Akun": username,
                "Peran": user_info.get("role", "-"),
                "Tanggal Buat": user_info.get("created_at", "-"),
                "Terkunci Sampai": datetime.fromtimestamp(user_info.get("lockout_until", 0)).strftime("%Y-%m-%d %H:%M:%S") if user_info.get("lockout_until", 0) > 0 else "-"
            })

        df_users = pd.DataFrame(user_data)
        st.dataframe(df_users[["Nama Akun", "Peran", "Tanggal Buat", "Terkunci Sampai"]], use_container_width=True)

        selected_user = st.selectbox("Pilih Pengguna", usernames)

        # Aksi admin untuk akun selain "admin"
        if selected_user != "admin":
            col1_reset, col2_delete = st.columns(2)
            with col1_reset:
                if st.button("Reset Kata Sandi Pengguna", key="reset_password_admin_direct"):
                    default_password = "11111111"
                    success, message = reset_password(selected_user, default_password)
                    if success:
                        st.success(f"Kata sandi untuk '{selected_user}' telah direset dengan kata sandi default.")
                    else:
                        st.error(message)
                    st.rerun()
            with col2_delete:
                if st.button("Hapus Pengguna", key="hapus_pengguna"):
                    delete_user(selected_user)
                    st.success(f"Pengguna '{selected_user}' dihapus.")
                    st.rerun()
            col3_change_role = st.columns(1)[0]
            with col3_change_role:
                new_role = st.selectbox("Ubah Peran Pengguna", ["user", "admin"], index=["user", "admin"].index(users[selected_user].get("role", "user")), key="select_new_role")
                if st.button("Ubah Peran", key="ubah_peran_pengguna"):
                    if new_role != users[selected_user].get("role"):
                        users_col.update_one({"username": selected_user}, {"$set": {"role": new_role}})
                        st.success(f"Peran untuk '{selected_user}' diubah menjadi '{new_role}'.")
                        st.rerun()
                    else:
                        st.info("Peran tidak berubah.")
        else:
            st.info("Akun admin tidak dapat direset, diubah peran, atau dihapus secara langsung di sini.")
    else:
        st.info("Belum ada pengguna yang terdaftar.")

    st.divider()
    
    st.markdown("#### 2. Permintaan Reset Kata Sandi dari Pengguna")

    pending_reset_requests = get_pending_password_reset_requests()

    if pending_reset_requests:
        request_data_reset = []
        for req in pending_reset_requests:
            request_data_reset.append({
                "ID Permintaan": str(req["_id"]),
                "Nama Akun": req["username"],
                "Tanggal Permintaan": req["request_date"]
            })

        df_requests_reset = pd.DataFrame(request_data_reset)
        st.dataframe(df_requests_reset, use_container_width=True)

        st.write("Pilih permintaan reset kata sandi untuk ditindaklanjuti:")

        request_options_reset = {f"{req['username']} ({req['request_date']})": req["_id"] for req in pending_reset_requests}
        selected_request_display_reset = st.selectbox("Pilih Permintaan Reset Kata Sandi", list(request_options_reset.keys()))
        selected_request_id_reset = request_options_reset[selected_request_display_reset]

        col1_reset, col2_reset = st.columns(2)
        with col1_reset:
            if st.button("Setujui & Reset Kata Sandi", key=f"approve_reset_password_{selected_request_id_reset}"):
                success, message = resolve_password_reset_request(selected_request_id_reset, action="reset")
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        with col2_reset:
            if st.button("Tolak Permintaan Reset", key=f"reject_reset_password_{selected_request_id_reset}"):
                success, message = resolve_password_reset_request(selected_request_id_reset, action="rejected")
                if success:
                    st.info(message)
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.info("Tidak ada permintaan reset kata sandi yang tertunda.")

    st.divider()

    st.markdown("#### 3. Permintaan Penghapusan Akun dari Pengguna")

    pending_delete_requests = get_pending_delete_requests()

    if pending_delete_requests:
        request_data_delete = []
        for req in pending_delete_requests:
            request_data_delete.append({
                "ID Permintaan": str(req["_id"]),
                "Nama Akun": req["username"],
                "Tanggal Permintaan": req["request_date"]
            })

        df_requests_delete = pd.DataFrame(request_data_delete)
        st.dataframe(df_requests_delete, use_container_width=True)

        st.write("Pilih permintaan penghapusan akun untuk ditindaklanjuti:")

        request_options_delete = {f"{req['username']} ({req['request_date']})": req["_id"] for req in pending_delete_requests}
        selected_request_display_delete = st.selectbox("Pilih Permintaan Penghapusan", list(request_options_delete.keys()))
        selected_request_id_delete = request_options_delete[selected_request_display_delete]

        col1_delete, col2_delete = st.columns(2)
        with col1_delete:
            if st.button("Setujui & Hapus Akun Ini", key=f"approve_delete_user_{selected_request_id_delete}"):
                success, message = resolve_delete_request(selected_request_id_delete, action="deleted")
                if success:
                    st.success(message)
                    st.rerun()
                else:
                    st.error(message)
        with col2_delete:
            if st.button("Tolak Permintaan Ini", key=f"reject_delete_user_{selected_request_id_delete}"):
                success, message = resolve_delete_request(selected_request_id_delete, action="rejected")
                if success:
                    st.info(message)
                    st.rerun()
                else:
                    st.error(message)
    else:
        st.info("Tidak ada permintaan penghapusan akun yang tertunda.")