# ===== Import Library & Modul Eksternal =====
import io
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
import streamlit as st
from datetime import datetime, date
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.callbacks import EarlyStopping

# ===== Import Modul Autentikasi, Model, dan Fungsi Pendukung =====
from utils.auth import authenticate, get_user_role, show_auth_sidebar
from utils.model import list_model, get_model_file, load_model_file, load_model_metadata_file, parse_date, get_training_config, save_info_model, delete_old_model, create_dataset, build_and_train_model, highlight_rows, get_future_dates, dataset_information_summary, generate_lstm_model_config, model_architecture_summary, show_prediction_results, extract_model_info, show_extract_model_info_for_home_page
from dotenv import load_dotenv

# Konfigurasi awal Streamlit
st.set_page_config(page_title="Prediksi Harga Saham", page_icon="assets/favicon.ico", layout="wide")
st.header("Analisis Prediksi Harga Saham Menggunakan Metode LSTM", divider="gray")

# Memuat konfigurasi
load_dotenv()
role = show_auth_sidebar()
is_admin = role == "admin" if role else False

# Reproducibility menggunakan Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# Callback EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Sidebar dengan input parameter dari pengguna
st.sidebar.header("Pengaturan Model")
ticker = st.sidebar.text_input("Masukkan Ticker Saham", "HLAG.DE", help="Ticker adalah kode singkat untuk mewakili suatu perusahaan di pasar saham. Anda dapat mencari ticker saham di https://finance.yahoo.com/")
start_date_str = st.sidebar.text_input("Tanggal Awal", value="2000/01/01", help="Format tanggal: YYYY/MM/DD atau YYYY-MM-DD.")
end_date_str = st.sidebar.text_input("Tanggal Akhir", value=date.today().strftime("%Y/%m/%d"), help="Format tanggal: YYYY/MM/DD atau YYYY-MM-DD.")
freq = st.sidebar.selectbox("Frekuensi Data", options=["Harian", "Mingguan", "Bulanan"], help="Frekuensi data merupakan frekuensi pengambilan data harga saham.")
model_option = st.sidebar.radio("Metode Pemodelan", ["Latih model baru", "Gunakan model dari database"], help="Pemilihan metode `Latih model baru` disarankan untuk prediksi terkini. Referensi untuk pilihan `Gunakan model dari database` dapat dilihat pada tabel di layar.")
tune_model = st.sidebar.checkbox("Aktifkan Model Tuning", value=True, help="Mengombinasikan parameter model tuning secara otomatis untuk analisis prediksi yang mendalam.")

# Konfigurasi ini memuat kode frekuensi, time_step, batch_size, dan epochs
config = get_training_config(freq)
# Ekstrak parameter dari konfigurasi
freq_code = config["code"]
time_step = config["time_step"]
batch_size = config["batch_size"]
epochs = config["epochs"]

# Memuat dari database jika model tersedia
# Tombol untuk memulai proses prediksi
can_start_prediction = True # Initialize to True, will be set to False if issues
if model_option == "Gunakan model dari database":
    try:
        # Parsing tanggal input dan mengambil data dummy untuk menentukan rentang data aktual
        parsed_start_date = parse_date(start_date_str).strftime("%Y-%m-%d")
        parsed_end_date = parse_date(end_date_str).strftime("%Y-%m-%d")

        # Cek ketersediaan model baseline dan tuning terbaik di database
        model_name_baseline_check = f"{ticker}_{freq}_{parsed_start_date}_{parsed_end_date}_baseline"
        model_baseline_exists = load_model_metadata_file(model_name_baseline_check) is not None

        model_best_tuning_check = f"{ticker}_{freq}_{parsed_start_date}_{parsed_end_date}_best tuning"
        model_best_tuning_exists = load_model_metadata_file(model_best_tuning_check) is not None

        # Menampilkan peringatan jika model tidak tersedia
        if not model_baseline_exists:
            st.sidebar.warning(f"Model baseline untuk ticker `{ticker}` dari {start_date_str} sampai {end_date_str} dengan frekuensi data `{freq}` tidak ditemukan di database.")
            can_start_prediction = False

        if tune_model and not model_best_tuning_exists:
            st.sidebar.warning(f"Model tuning terbaik untuk ticker `{ticker}` dari {start_date_str} sampai {end_date_str} dengan frekuensi data `{freq}` tidak ditemukan di database. Salah satu cara untuk melanjutkan adalah menonaktifkan fitur model tuning.")
            can_start_prediction = False

        if not model_baseline_exists and (not tune_model or not model_best_tuning_exists):
            st.sidebar.info(f"Silakan ganti parameter atau pilih `Latih model baru` jika Anda ingin melanjutkan.")
            can_start_prediction = False

        if model_baseline_exists:
            if tune_model and model_best_tuning_exists:
                st.sidebar.success(f"Model tuning terbaik untuk ticker `{ticker}` dari {start_date_str} sampai {end_date_str} dengan frekuensi data `{freq}` ditemukan di database.")
            st.sidebar.success(f"Model baseline untuk ticker `{ticker}` dari {start_date_str} sampai {end_date_str} dengan frekuensi data `{freq}` ditemukan di database.")

    except ValueError as e:
        st.sidebar.error(f"Terjadi kesalahan dalam format tanggal: {e}")
        can_start_prediction = False
    except Exception as e:
        st.sidebar.error(f"Terjadi kesalahan saat memeriksa model di database: {e}")
        can_start_prediction = False

# Tombol untuk memulai proses prediksi
if can_start_prediction:
    start_button_pressed = st.sidebar.button("Mulai Prediksi")
else:
    st.sidebar.button("Mulai Prediksi", disabled=True)
    start_button_pressed = False

# Menekan tombol Ditekan untuk mulai proses
if start_button_pressed:
    # Business Understanding
    st.subheader("1. Business Understanding")
    st.markdown(
        """
        <div style='text-align: justify; margin-bottom: 10px'>
            Dalam dunia investasi dan pasar modal, kemampuan untuk memprediksi harga saham secara akurat 
            sangat penting bagi pengambilan keputusan yang tepat dan strategis. Oleh karena itu, dibutuhkan 
            metode yang mampu menangkap pola waktu (time series) secara efektif. Long Short-Term Memory (LSTM) 
            dirancang untuk mengenali pola dalam data berurutan dan memiliki keunggulan dalam mengatasi masalah 
            long-term dependencies. Dengan menerapkan LSTM, perusahaan atau investor dapat memeroleh prediksi 
            harga saham yang mendukung perencanaan dan manajemen risiko investasi.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    # 2. Data Understanding
    st.subheader("2. Data Understanding")
    try:
        start_date = parse_date(start_date_str)
        end_date = parse_date(end_date_str)

        if start_date > end_date:
            st.sidebar.error("Tanggal Awal tidak boleh lebih besar dari Tanggal Akhir.")
            st.stop()
    except ValueError as e:
        st.sidebar.error(str(e))
        st.stop()

    # Unduh data dari Yahoo Finance
    with st.spinner("Mengambil data dari Yahoo Finance..."):
        if start_date >= end_date:
            st.sidebar.error("Tanggal awal harus lebih kecil dari tanggal akhir.")
            st.stop()

        end_date_for_yf = (datetime.combine(end_date, datetime.min.time()) + pd.Timedelta(days=1)).date()

        data = yf.download(ticker, start=start_date, end=end_date_for_yf, interval=freq_code, auto_adjust=True)

        if data.empty:
            st.error(f"Data dengan ticker {ticker} tidak ditemukan atau gagal diunduh.")
        else:
            actual_start_date = data.index.min().to_pydatetime().date()
            actual_end_date = data.index.max().to_pydatetime().date()

            # Deskripsi awal dataset
            st.markdown(
                f"""
                <div style='text-align: justify; margin-bottom: 10px'>
                    Data dengan ticker <strong>{ticker}</strong> dari tanggal 
                    <strong>{actual_start_date.strftime('%d %b, %Y')}</strong> sampai 
                    <strong>{actual_end_date.strftime('%d %b, %Y')}</strong> berhasil diunduh dari Yahoo Finance.
                    Dataset dapat dilihat pada tautan 
                    <a href='https://finance.yahoo.com/quote/{ticker}/history' target='_blank'>ini</a>.
                    Data understanding dilakukan untuk mengenali karakteristik data historis saham seperti kolom 
                    pada tabel data yang dikumpulkan. Tahap ini membantu memahami 
                    bagaimana pola harga saham berubah dari waktu ke waktu.
                </div>
                """,
                unsafe_allow_html=True
            )

    # Menampilkan ringkasan informasi dataset mentah
    dataset_information_summary(data, "Dataset mentah saham", expanded=True)

    st.divider()

    # Data Preparation
    st.subheader("3. Data Preparation")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Data diproses dan dibersihkan dengan cara nilai harga diubah ke dalam skala yang seragam agar model lebih mudah belajar. Kami juga mengatur data dalam format yang cocok untuk model LSTM (Long Short-Term Memory), yang merupakan jenis jaringan saraf tiruan khusus untuk data berurutan seperti harga saham harian atau mingguan.</div>""", unsafe_allow_html=True)

    st.markdown("#### 3.1 Data Transformation")
    # Menampilkan informasi bahwa data berhasil dimuat
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Memastikan data berhasil dimuat agar bisa diproses lebih lanjut.</div>""", unsafe_allow_html=True)

    # Mengubah menjadi single index jika kolomnya adalah MultiIndex
    with st.expander("Mengganti multiIndex dengan indeks biasa."):
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            data.columns.name = None
        st.write(data.head())
    
    # Reset index agar 'Date' menjadi kolom biasa dan ubah menjadi tipe datetime
    with st.expander("Mengubah indeks Date menjadi feature dan mengonversi tipe datanya ke datetime."):
        data.reset_index(inplace=True)
        data["Date"] = pd.to_datetime(data["Date"])
        st.write(data.head())
    
    st.markdown("#### 3.2 Data Filtering")
    # Menjelaskan tujuan filter data
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Menyesuaikan data sesuai kebutuhan analisis dengan memilih kode saham (ticker) dan frekuensi waktu (harian, mingguan, atau bulanan).</div>""", unsafe_allow_html=True)

    # Filter hanya kolom 'Date' dan 'Close' dan bulatkan harga
    df = data[["Date", "Close"]].copy()
    df["Close"] = df["Close"].round(2)

    # Simpan tanggal awal dan akhir data
    actual_start_date = df["Date"].min().strftime("%Y-%m-%d")
    actual_end_date = df["Date"].max().strftime("%Y-%m-%d")

    # Menampilkan ringkasan data yang sudah difilter
    dataset_information_summary(df, "Data yang difilter", expanded=True)

    st.subheader("3.3 Data Cleaning")
    # Menjelaskan pentingnya data bersih
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Pada tahap ini, data diperiksa dan dibersihkan dari masalah yang bisa mengganggu proses pelatihan model.</div>""", unsafe_allow_html=True)

    st.markdown("##### 3.3.1 Missing Values")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Jika ada tanggal tertentu tanpa data harga saham, data tersebut akan dihapus atau diisi agar tidak memengaruhi hasil model.</div>""", unsafe_allow_html=True)
    with st.expander("Memeriksa dan menangani nilai yang hilang pada feature Close."):
        missing_before = df[["Close"]].isnull().sum()

        # Isi missing value dengan metode forward fill jika ada
        if df["Close"].isnull().sum() > 0:
            df["Close"] = df["Close"].fillna(method="ffill")

        missing_after = df[["Close"]].isnull().sum()

        # Tampilkan ringkasan missing values sebelum dan sesudah ditangani
        missing_data = {
            "Feature": ["Close"],
            "Missing Values": missing_before.values,
            "Keterangan": [
                "Tidak ada missing value" if missing_before["Close"] == 0 else "Terdapat missing value"
            ]
        }

        if missing_before["Close"] > 0:
            missing_data["Missing sesudah Ditangani"] = missing_after.values
        missing_table = pd.DataFrame(missing_data)
        st.write(missing_table)
    
    st.markdown("##### 3.3.2 Negative Values")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Melihat nilai data yang negatif dan tidak memengaruhi hasil model.</div>""", unsafe_allow_html=True)
    # Periksa apakah ada nilai negatif
    with st.expander("Memeriksa nilai yang negatif pada feature Close."):
        negative_table = pd.DataFrame({
            "Feature": ["Close"],
            "Negative Values": [(df["Close"] < 0).sum()],
            "Keterangan": [
                "Tidak ada nilai negatif" if (df["Close"] < 0).sum() == 0 else "Terdapat nilai negatif"
            ]
        })
        st.write(negative_table)

    st.markdown("#### 3.4 Exploratory Data Analysis (EDA)")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>EDA adalah proses melihat dan memahami pola dalam data secara visual sebelum membangun model.</div>""", unsafe_allow_html=True)

    st.markdown("##### 3.4.1 Ringkasan statistik pada harga saham.")
    # Ringkasan statistik harga saham
    st.write(df["Close"].describe())
    df["Year"] = df["Date"].dt.year
    year_counts = df["Year"].value_counts().sort_index()

    st.markdown("##### 3.4.2 Melihat distribusi nilai harga.")
    # Distribusi jumlah data
    fig, ax = plt.subplots(figsize=(10, 5))
    # Visualisasi distribusi data per tahun
    bars = ax.bar(year_counts.index.astype(str), year_counts.values, color="skyblue")

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold"
        )

    ax.set_xlabel("Tahun", fontsize=12)
    ax.set_ylabel("Jumlah Data", fontsize=12)
    ax.set_title("Distribusi Jumlah Data per Tahun", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    ax.tick_params(axis='x', labelrotation=45)
    st.pyplot(fig)

    st.markdown("##### 3.4.3 Menampilkan grafik harga saham dari waktu ke waktu.")
    # Grafik harga saham dari waktu ke waktu
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], color="blue")
    ax.set_title("Grafik Harga Saham", fontsize=14)
    ax.set_xlabel("Tahun", fontsize=12)
    ax.set_ylabel("Harga", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("#### 3.5 Normalization")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Harga saham diubah ke dalam skala antara 0 sampai 1 (normalisasi) menggunakan teknik yang disebut MinMaxScaler.
            Ini dilakukan agar model lebih mudah mempelajari pola di mana sebelumnya angka-angka besar bisa menyulitkan model.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Normalisasi menggunakan MinMaxScaler
    scaler = MinMaxScaler()
    df["Close Scaled"] = scaler.fit_transform(df[["Close"]])

    # Tampilkan hasil normalisasi
    with st.expander("Data yang telah dinormalisasi tetapi belum disusun sebagai suatu sequence (urutan)"):
        df_normalized = pd.DataFrame(df[["Date", "Close Scaled"]])
        df_normalized = df_normalized.rename(columns={"Date": "Tanggal", "Close Scaled": "Close yang Diskalasikan"})
        st.dataframe(df_normalized)
        st.write(f"Ukuran tabel: {df_normalized.shape}")

    st.markdown("#### 3.6 Data Windowing")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Model tidak hanya melihat satu harga sebelumnya untuk memprediksi harga berikutnya, 
            tapi beberapa hari/minggu/bulan ke belakang, yang disebut time step. Misalnya, 
            jika time step = 30 maka model akan melihat 30 hari sebelumnya untuk memprediksi 
            hari ke-31. Hal ini dilakukan untuk menyediakan pola urutan agar model bisa mempelajari tren waktu.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Bentuk urutan data berdasarkan time step (misal 30 hari sebelumnya untuk prediksi hari ke-31)
    data_scaled = df["Close Scaled"].values.reshape(-1, 1)
    X_seq, y_seq = create_dataset(data_scaled, time_step)

    # Tampilkan hasil windowing
    df_seq = pd.DataFrame({
        "X (sequence)": [x.flatten().round(4).tolist() for x in X_seq],
        "y (target)": y_seq.flatten().round(4)
    })
    with st.expander("Data yang telah dinormalisasi dan disusun sebagai sequence untuk input (X) dan target (y)"):
        st.dataframe(df_seq)
        st.write(f"Ukuran tabel: {df_seq.shape}")

    st.markdown("#### 3.7 Data Splitting")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Data dibagi menjadi dua, yaitu data latih (80%) untuk mempelajari model pada pola masa lalu dan data uji (20%) untuk menguji kemampuan model pada data yang belum pernah dilihat. Pembagian dapat menilai model dalam membuat prediksi yang akurat terhadap data baru.</div>""",unsafe_allow_html=True)
    # Split data ke train dan test set
    X, y = create_dataset(data_scaled, time_step)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    # Mengubah ke format [samples, timesteps, features] untuk LSTM
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Plot garis pembatas antara data latih dan uji
    full_data = np.concatenate([y_train, y_test])
    full_data_rescaled = scaler.inverse_transform(full_data.reshape(-1, 1))
    plot_dates = df["Date"].iloc[time_step:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_dates, full_data_rescaled, label="Harga", color="blue")

    split_date = plot_dates.iloc[split]
    # Garis split train/test
    ax.axvline(split_date, color="red", linestyle="--", label="Train/Test Split")

    ax.fill_between(plot_dates[:split], full_data_rescaled[:split, 0], color="lightblue", alpha=0.5, label="Train")
    ax.fill_between(plot_dates[split:], full_data_rescaled[split:, 0], color="orange", alpha=0.5, label="Test")

    ax.set_title("Pembagian Data Train dan Test")
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    df_X_and_y_train = pd.DataFrame({
        "X_train (sequence)": X_train[i].flatten().round(4).tolist(),
        "y_train": round(float(y_train[i]), 4)
    } for i in range(len(X_train)))
    # Tampilkan isi dari X_train dan y_train
    with st.expander("Kelompok data X_train dan y_train"):
        st.dataframe(df_X_and_y_train)
        st.write(f"Ukuran tabel: {df_X_and_y_train.shape}")

    df_X_and_y_test = pd.DataFrame({
        "X_test (sequence)": X_test[i].flatten().round(4).tolist(),
        "y_test": round(float(y_test[i]), 4)
    } for i in range(len(X_test)))
    # Tampilkan isi dari X_test dan y_test
    with st.expander("Kelompok data X_test dan y_test"):
        st.dataframe(df_X_and_y_test)
        st.write(f"Ukuran tabel: {df_X_and_y_test.shape}")

    st.divider()

    st.subheader("4. Modeling")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Model baseline (dasar) dilatih terlebih dahulu dengan parameter awal. 
            Apabila memerlukan proses tuning (melakukan berbagai kombinasi parameter untuk 
            mencari hasil prediksi terbaik) maka membutuhkan lebih banyak waktu karena 
            setiap model diuji satu per satu. Hasilnya dibandingkan berdasarkan akurasi prediksi 
            menggunakan metrik seperti RMSE (Root Mean Squared Error) dan MAPE (Mean Absolute Percentage Error).
        </div>
        """,
        unsafe_allow_html=True
    )
    model_baseline_loaded = None
    history_baseline_loaded = None
    duration_baseline_loaded = None
    epochs_baseline_loaded = None
    epochs_trained_baseline_loaded = None
    batch_size_baseline_loaded = None

    # Pilihan menggunakan model dari database
    if model_option == "Gunakan model dari database":
        model_name_baseline = f"{ticker}_{freq}_{actual_start_date}_{actual_end_date}_baseline"
        try:
            with st.spinner(f"Memuat model baseline dan metadata dari database untuk `{model_name_baseline}`..."):
                # Coba load model dan metadata
                model_baseline_loaded = load_model_file(model_name_baseline)
                metadata_baseline_loaded = load_model_metadata_file(model_name_baseline)
                if model_baseline_loaded and metadata_baseline_loaded:
                    history_baseline_loaded = tf.keras.callbacks.History()
                    history_baseline_loaded.history = metadata_baseline_loaded.get("history", {})

                    epochs_baseline_loaded = metadata_baseline_loaded.get("epochs")
                    epochs_trained_baseline_loaded = metadata_baseline_loaded.get("epochs_trained")
                    batch_size_baseline_loaded = metadata_baseline_loaded.get("batch_size")
                    duration_baseline_loaded = metadata_baseline_loaded.get("duration")
                    rmse_baseline_loaded = metadata_baseline_loaded.get("rmse")
                    mape_baseline_loaded = metadata_baseline_loaded.get("mape")
                        
                    st.success(f"Model baseline dan metadata untuk '{model_name_baseline}' berhasil dimuat dari database.")
                        
                    model = model_baseline_loaded
                    history = history_baseline_loaded
                    epochs = epochs_baseline_loaded
                    epochs_trained = epochs_trained_baseline_loaded
                    batch_size = batch_size_baseline_loaded
                    duration = duration_baseline_loaded
                else:
                    # Melatih model baru jika gagal
                    st.warning(f"Model baseline atau metadata untuk `{model_name_baseline}` tidak ditemukan di database. Melatih model baru sebagai gantinya.")
                    model, history, duration, epochs, epochs_trained, batch_size = build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs, batch_size, callbacks=[early_stop])
        except Exception as e:
            # Tangani error saat loading
            st.error(f"Gagal memuat model baseline dari database: {e}. Melatih model baru sebagai gantinya.")
            model, history, duration, epochs, epochs_trained, batch_size = build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs, batch_size, callbacks=[early_stop])
    else:
        # Opsi lain dengan melatih model dari awal
        with st.spinner("Melatih model dan melakukan prediksi..."):
            model, history, duration, epochs, epochs_trained, batch_size = build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs, batch_size, callbacks=[early_stop])

    st.markdown("#### 4.1 Model Selection")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Pelatihan model menggunakan metode LSTM dengan menghasilkan satu model baseline.</div>""", unsafe_allow_html=True)
    # Tampilkan konfigurasi model
    generate_lstm_model_config(model=model, time_step=time_step, epochs=epochs, epochs_trained=epochs_trained, batch_size=batch_size, title="4.1.1 Model Configuration")

    st.markdown("##### 4.1.2 Model Architecture Summary")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Struktur pada model baseline dapat dilihat sebagai berikut.</div>""", unsafe_allow_html=True)
    # Tampilkan arsitektur model
    model_architecture_summary(model)

    st.markdown("#### 4.2 Model Training")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Menampilkan pengaturan (parameter) dari model yang digunakan, seperti time step (jumlah langkah waktu), jumlah epoch (berapa kali model mempelajari seluruh data), dan ukuran batch (berapa data yang diproses sekaligus saat pelatihan).</div>""", unsafe_allow_html=True)
    # Plot loss training dan validasi
    fig, ax = plt.subplots(figsize=(8, 4))
    if history is not None and history.history:
        ax.plot(history.history["loss"], label="Training Loss", color="blue")
        if "val_loss" in history.history:
            ax.plot(history.history["val_loss"], label="Validation Loss", color="orange")
    else:
        st.info("History tidak ditemukan atau gagal dimuat dari file.")

    ax.set_title("Visualisasi Training Loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.grid(True)
    ax.legend()
    st.pyplot(fig)

    # Lakukan prediksi dan inverskan skala ke nilai aslinya
    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Menampilkan hasil prediksi dari model baseline
    st.markdown("#### 4.3 Model Inference")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Model yang sudah dilatih digunakan untuk memprediksi harga saham pada data 
            yang belum pernah dilihat sebelumnya (data uji). Hasil prediksi dibandingkan 
            dengan data asli untuk menilai seberapa akurat model memprediksi. Ini membantu 
            menentukan apakah model sudah cukup baik atau masih perlu ditingkatkan.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Hasil prediksi baseline
    st.markdown("##### 4.3.1 Prediction Results")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Menampilkan hasil prediksi pada model baseline.</div>""", unsafe_allow_html=True)

    # Menentukan tanggal untuk plot hasil prediksi
    plot_dates_test = df["Date"].iloc[time_step + len(y_train):]

    # Fungsi custom menampilkan hasil prediksi dan menghitung metrik evaluasi
    df_result, rmse, mape = show_prediction_results(
        y_true_rescaled=y_test_rescaled,
        y_pred_rescaled=y_pred_rescaled,
        plot_dates=plot_dates_test,
        title="Hasil Prediksi"
    )

    # Menyimpan informasi hasil baseline
    baseline_type = "Baseline (dari Database)" if model_option == "Gunakan model dari database" else "Baseline"
    baseline_results = [{
        "time_step": time_step,
        "epochs": epochs,
        "epochs_trained": epochs_trained,
        "batch_size": batch_size,
        "rmse": rmse,
        "mape": mape,
        "model": model,
        "history": history,
        "duration": duration,
        "tipe": baseline_type
    }]

    # Jika melatih model baru, simpan model dan metadata ke database
    if model_option == "Latih model baru":
        metadata = {
            "time_step": time_step,
            "epochs": epochs,
            "epochs_trained": epochs_trained,
            "batch_size": batch_size,
            "rmse": rmse,
            "mape": mape,
            "duration": duration,
        }
        files_deleted = delete_old_model(freq, ticker, actual_start_date, actual_end_date, model_type="baseline")
        save_info_model(model, freq, ticker=ticker, start_date=actual_start_date, end_date=actual_end_date, model_type="baseline", history=history, metadata=metadata)
        if not st.session_state.logged_in:
            st.info("Silahkan masuk untuk dapat melihat dashboard model dan metadata.")

    # Membuat dan menampilkan tabel hasil baseline
    baseline_table = pd.DataFrame([{
        "Tipe Model": baseline_type,
        "Time Step": time_step,
        "Epoch": epochs,
        "Epochs Terlatih": epochs_trained,
        "Batch Size": batch_size,
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape * 100, 2),
        "Durasi (detik)": round(duration, 2)
    }])
    baseline_table = (baseline_table.style.format({"RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "Durasi (detik)": "{:.2f}"}))

    # Proses hyperparameter tuning
    if tune_model:
        st.markdown("##### 4.3.2 Model Results")
        st.markdown(
            f"""
            <div style='text-align: justify; margin-bottom: 10px'>
                Menampilkan hasil evaluasi model baseline dalam bentuk tabel. Terdapat metrik RMSE
                di mana semakin kecil nilainya maka semakin akurat prediksi. Adapun metrik MAPE
                dengan persentase kesalahan rata-rata yang semakin kecil nilainya maka semakin baik.
            </div>
            """,
            unsafe_allow_html=True
        )
        st.dataframe(baseline_table)

        st.markdown("#### 4.4 Model Tuning")
        st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Apabila mengaktifkan model tuning maka hyperparameter tuning dilakukan dengan beberapa kombinasi pengaturan (time step, epoch, dan batch size) untuk mencari model terbaik.</div>""", unsafe_allow_html=True)

        # Load model tuning terbaik dari database jika ada
        if model_option == "Gunakan model dari database":
            tuning_results = []
            model_name_best_tuning = f"{ticker}_{freq}_{actual_start_date}_{actual_end_date}_best tuning"
            try:
                with st.spinner(f"Memuat model best tuning dan metadata dari database untuk `{model_name_best_tuning}`..."):
                    model_best_tuning_loaded = load_model_file(model_name_best_tuning)
                    metadata_best_tuning_loaded = load_model_metadata_file(model_name_best_tuning)

                    if model_best_tuning_loaded and metadata_best_tuning_loaded:
                        history_best_tuning_loaded = tf.keras.callbacks.History()
                        history_best_tuning_loaded.history = metadata_best_tuning_loaded.get("history", {})

                        # Tambahkan ke tuning_results jika model tuning berhasil dimuat
                        tuning_results.append({
                            "time_step": metadata_best_tuning_loaded.get("time_step"),
                            "epochs": metadata_best_tuning_loaded.get("epochs"),
                            "epochs_trained": metadata_best_tuning_loaded.get("epochs_trained"),
                            "batch_size": metadata_best_tuning_loaded.get("batch_size"),
                            "rmse": metadata_best_tuning_loaded.get("rmse"),
                            "mape": metadata_best_tuning_loaded.get("mape"),
                            "model": model_best_tuning_loaded,
                            "history": history_best_tuning_loaded,
                            "duration": metadata_best_tuning_loaded.get("duration"),
                            "tipe": "Best Tuning (dari Database)"
                        })
                        st.success(f"Model best tuning dan metadata untuk '{model_name_best_tuning}' berhasil dimuat dari database.")
                    else:
                        st.warning(f"Model best tuning atau metadata untuk '{model_name_best_tuning}' tidak ditemukan di database. Akan melakukan tuning model baru jika opsi 'Latih model baru' dipilih.")
            except Exception as e:
                st.error(f"Gagal memuat model best tuning dari database: {e}.")
            best = sorted(tuning_results, key=lambda x: x["rmse"])[0]
            metadata_best = {
                "time_step": best["time_step"],
                "epochs": best["epochs"],
                "epochs_trained": best["epochs_trained"],
                "batch_size": best["batch_size"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "duration": best["duration"],
            }
        
        # Melakukan tuning dengan kombinasi hyperparameter jika model tuning belum tersedia
        if model_option == "Latih model baru" or (model_option == "Gunakan model dari database" and model_best_tuning_loaded is None):
            if freq == "Harian":
                time_steps_list = [30, 60]
                epochs_list = [50, 75, 100]
                batch_sizes_list = [32, 64]
            elif freq == "Mingguan":
                time_steps_list = [12, 24]
                epochs_list = [75, 100, 125]
                batch_sizes_list = [8, 16]
            else:
                time_steps_list = [6, 12]
                epochs_list = [100, 125, 150]
                batch_sizes_list = [2, 4]

            # Simpan parameter baseline untuk pengecualian di tuning
            baseline_time_step = time_step
            baseline_epochs = epochs
            baseline_batch_size = batch_size
            tuning_combinations = []
            
            for ts in time_steps_list:
                for ep in epochs_list:
                    for bs in batch_sizes_list:
                        if not (ts == baseline_time_step and ep == baseline_epochs and bs == baseline_batch_size):
                            tuning_combinations.append((ts, ep, bs))

            total_iter = len(tuning_combinations)
            progress_bar = st.progress(0)
            status_text = st.empty()
            tuning_results = []
            
            iter_count = 0
            for ts, ep, bs in tuning_combinations:
                # Melatih model dengan kombinasi hyperparameter
                iter_count += 1
                status_text.text(f"Melatih model dengan time step = {ts}, epoch = {ep}, dan batch size = {bs} ({iter_count}/{total_iter})")

                X, y = create_dataset(data_scaled, ts)
                split = int(len(X)*0.8)
                X_train_tune, X_test_tune = X[:split], X[split:]
                y_train_tune, y_test_tune = y[:split], y[split:]

                X_train_tune = X_train_tune.reshape(X_train_tune.shape[0], X_train_tune.shape[1], 1)
                X_test_tune = X_test_tune.reshape(X_test_tune.shape[0], X_test_tune.shape[1], 1)

                model_temp, history_temp, duration_temp, _, epochs_trained, _ = build_and_train_model(X_train_tune, y_train_tune, X_test_tune, y_test_tune, ts, epochs=ep, batch_size=bs, callbacks=[early_stop])
                pred = model_temp.predict(X_test_tune)
                pred = scaler.inverse_transform(pred)
                actual = scaler.inverse_transform(y_test_tune.reshape(-1, 1))
                rmse = math.sqrt(mean_squared_error(actual, pred))
                mape = mean_absolute_percentage_error(actual, pred)

                tuning_results.append({
                    "time_step": ts,
                    "epochs": ep,
                    "epochs_trained": epochs_trained,
                    "batch_size": bs,
                    "rmse": rmse,
                    "mape": mape,
                    "model": model_temp,
                    "history": history_temp,
                    "duration": duration_temp,
                    "tipe": "Tuning"
                })

                progress_bar.progress(iter_count / total_iter)

        # Menyimpan model tuning terbaik ke database
        if model_option == "Latih model baru":
            best = sorted(tuning_results, key=lambda x: x["rmse"])[0]
            metadata_best = {
                "time_step": best["time_step"],
                "epochs": best["epochs"],
                "epochs_trained": best["epochs_trained"],
                "batch_size": best["batch_size"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "duration": best["duration"],
            }
            files_deleted = delete_old_model(freq, ticker, actual_start_date, actual_end_date, model_type="best tuning")
            save_info_model(best["model"], freq, ticker=ticker, start_date=actual_start_date, end_date=actual_end_date, model_type="best tuning", history=best["history"], metadata=metadata_best)

            if not st.session_state.logged_in:
                st.info("Silahkan masuk untuk dapat melihat dashboard model dan metadata.")

        # Menampilkan informasi model terbaik
        st.markdown("#### 4.5 Best Model Selection")
        st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Pemilihan dilakukan kepada satu model terbaik berdasarkan hasil evaluasi paling akurat (nilai RMSE terendah) dari semua model yang dicoba selama tuning.</div>""", unsafe_allow_html=True)
        st.success(f"Model terbaik menggunakan parameter time step = {best['time_step']}, epoch = {best['epochs']}, dan batch size = {best['batch_size']} dengan metrik evaluasi RMSE sebesar {best['rmse']:.4f} dan MAPE sebesar {best['mape']:.2%}")
        best_model = best["model"]
        best_history = best["history"]
        best_time_step = best["time_step"]
        best_epochs = best["epochs"]
        best_epochs_trained = best["epochs_trained"]
        best_batch_size = best["batch_size"]
        best_duration = best["duration"]

        # Konfigurasi dan arsitektur model terbaik
        generate_lstm_model_config(model=best["model"], time_step=best["time_step"], epochs=best["epochs"], epochs_trained=best["epochs_trained"], batch_size=best["batch_size"], title="4.5.1 Best Model Configuration")

        st.markdown("##### 4.5.2 Best Model Architecture Summary")
        st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Struktur pada model best tuning dapat dilihat sebagai berikut.</div>""", unsafe_allow_html=True)
        model_architecture_summary(best_model)

        # Visualisasi loss saat training pada model terbaik
        st.markdown("#### 4.6 Best Model Training")
        st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Menampilkan pengaturan (parameter) dari model yang digunakan, seperti time step (jumlah langkah waktu), jumlah epoch (berapa kali model mempelajari seluruh data), dan ukuran batch (berapa data yang diproses sekaligus saat pelatihan).</div>""", unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(8, 4))
        if best["history"] is not None and best["history"].history:
            ax.plot(best["history"].history["loss"], label="Training Loss", color="blue")
            if "val_loss" in best["history"].history:
                ax.plot(best["history"].history["val_loss"], label="Validation Loss", color="orange")
        else:
            st.info("History tidak ditemukan atau gagal dimuat dari file.")

        ax.set_title("Visualisasi Training Loss pada Model Terbaik")
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Loss")
        ax.grid(True)
        ax.legend()
        st.pyplot(fig)

        # Menampilkan prediksi dan evaluasi dari model terbaik
        st.markdown("#### 4.7 Best Model Inference")
        st.markdown(
            f"""
            <div style='text-align: justify; margin-bottom: 10px'>
                Model yang sudah dilatih digunakan untuk memprediksi harga saham pada data 
                yang belum pernah dilihat sebelumnya (data uji). Hasil prediksi dibandingkan 
                dengan data asli untuk menilai seberapa akurat model memprediksi. Ini membantu 
                menentukan apakah model sudah cukup baik atau masih perlu ditingkatkan.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Hasil prediksi model tuning terbaik
        st.markdown("##### 4.7.1 Best Model Prediction Results")
        st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Menampilkan hasil prediksi pada model tuning.</div>""", unsafe_allow_html=True)
        X, y = create_dataset(data_scaled, best["time_step"])
        split = int(len(X) * 0.8)
        X_train_best_model, X_test_best_model = X[:split], X[split:]
        y_train_best_model, y_test_best_model = y[:split], y[split:]

        X_train_best_model = X_train_best_model.reshape(X_train_best_model.shape[0], X_train_best_model.shape[1], 1)
        X_test_best_model = X_test_best_model.reshape(X_test_best_model.shape[0], X_test_best_model.shape[1], 1)

        y_pred_best = best["model"].predict(X_test_best_model)
        y_pred_best_rescaled = scaler.inverse_transform(y_pred_best)
        y_test_best_model_rescaled = scaler.inverse_transform(y_test_best_model.reshape(-1, 1))

        plot_dates_best = df["Date"].iloc[best["time_step"] + len(y_train_best_model):]

        df_best_result, rmse_best, mape_best = show_prediction_results(
            y_true_rescaled=y_test_best_model_rescaled,
            y_pred_rescaled=y_pred_best_rescaled,
            plot_dates=plot_dates_best,
            title="Hasil Prediksi Model Terbaik"
        )

        # Hasil evaluasi model tuning terbaik
        st.markdown("##### 4.7.2 Best Model Results")
        st.markdown(
            f"""
            <div style='text-align: justify; margin-bottom: 10px'>
                Menampilkan hasil evaluasi model baseline dalam bentuk tabel. Terdapat metrik RMSE
                di mana semakin kecil nilainya maka semakin akurat prediksi. Adapun metrik MAPE
                dengan persentase kesalahan rata-rata yang semakin kecil nilainya maka semakin baik.
            </div>
            """,
            unsafe_allow_html=True
        )
        best_tuning_table = pd.DataFrame([{
            "Tipe Model": "Tuning terbaik",
            "Time Step": best["time_step"],
            "Epoch": best["epochs"],
            "Epochs Terlatih": best["epochs_trained"],
            "Batch Size": best["batch_size"],
            "RMSE": round(rmse_best, 4),
            "MAPE (%)": round(mape_best * 100, 2),
            "Durasi (detik)": round(best['duration'], 2)
        }])
        best_tuning_table = (best_tuning_table.style.format({"RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "Durasi (detik)": "{:.2f}"}))
        st.dataframe(best_tuning_table)

    st.divider()

    # Menampilkan ringkasan hasil evaluasi model baseline dan/atau tuning
    st.subheader("5. Evaluation")
    st.markdown(f"""<div style='text-align: justify; margin-bottom: 10px'>Evaluasi model ditampilkan untuk memudahkan pengguna dalam melihat prediksi model.</div>""", unsafe_allow_html=True)

    st.markdown("#### 5.1 Final Model Evaluation")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Menampilkan hasil evaluasi model final dalam bentuk tabel. Terdapat metrik RMSE
            di mana semakin kecil nilainya maka semakin akurat prediksi. Adapun metrik MAPE
            dengan persentase kesalahan rata-rata yang semakin kecil nilainya maka semakin baik. 
            Warna latar baris biru dan hijau pada baris tabel masing-masing menandakan model 
            baseline dan best tuning.
        </div>
        """,
        unsafe_allow_html=True
    )

    # Menggabungkan hasil baseline dan tuning jika tuning aktif
    if tune_model:
        all_results = baseline_results + tuning_results
        model_table = pd.DataFrame([{
            "Tipe Model": r["tipe"],
            "Time Step": r["time_step"],
            "Epoch": r["epochs"],
            "Epochs Terlatih": r["epochs_trained"],
            "Batch Size": r["batch_size"],
            "RMSE": round(r["rmse"], 4),
            "MAPE (%)": round(r['mape'] * 100, 2),
            "Durasi (detik)": round(r['duration'], 2)
        } for r in all_results])

        min_rmse_value = model_table["RMSE"].min()
        model_table = (model_table.style.apply(lambda row: highlight_rows(row, min_rmse_value), axis=1).format({"RMSE": "{:.4f}", "MAPE (%)": "{:.2f}", "Durasi (detik)": "{:.2f}"}))
        st.dataframe(model_table, use_container_width=True)
    else:
        st.dataframe(baseline_table)

    # Melakukan prediksi harga saham untuk n periode ke depan
    st.markdown("#### 5.2 Forecasting")
    unit = {"Harian": "Hari", "Mingguan": "Minggu", "Bulanan": "Bulan"}[freq]
    n_future = 10
    st.info(
        "Forecasting ini bersifat estimasi dan tidak merepresentasikan kejadian nyata secara pasti. "
        "Model menghasilkan proyeksi berdasarkan pola historis dan bukan kejadian masa depan yang pasti.\n\n"
        f"Model yang digunakan adalah {'hasil tuning' if tune_model else 'baseline'} untuk memprediksi {n_future} {unit.lower()} ke depan."
        f"Jumlah waktu sebesar {n_future} {unit.lower()} ke depan dipilih karena merupakan angka psikologis "
        "umum yang digunakan untuk memberikan gambaran ringkas. Angka ini cukup untuk menunjukkan tren "
        "yang mungkin terjadi."
    )

    # Model yang digunakan untuk prediksi masa depan
    future_model = best["model"] if tune_model else model
    future_time_step = best["time_step"] if tune_model else time_step

    last_input = data_scaled[-future_time_step:]
    temp_input = list(last_input.flatten())

    progress_bar = st.progress(0)
    status_text = st.empty()
    predictions = []

    # Iterasi memprediksi harga ke depan
    for i in range(n_future):
        status_text.text(f"Memproses iterasi ke-{i+1} dari {n_future}")
        x_input = np.array(temp_input[-future_time_step:]).reshape(1, future_time_step, 1)
        yhat = future_model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        predictions.append(yhat[0][0])
        progress_bar.progress((i + 1) / n_future)

    start_date = df["Date"].max()
    future_dates = get_future_dates(start_date, n_future, freq=freq)

    future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    df_future = pd.DataFrame({"Tanggal": future_dates, "Harga yang Diprediksi": future_prices.flatten()})

    # Membuat DataFrame hasil prediksi ke depan dan visualisasi
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_future["Tanggal"], df_future["Harga yang Diprediksi"], marker="o", linestyle="-")

    for i, (x, y) in enumerate(zip(df_future["Tanggal"], df_future["Harga yang Diprediksi"])):
        ax.text(
            x, y, 
            f"{y:.2f}",
            ha="center", 
            va="bottom", 
            fontsize=9, 
            rotation=0
        )

    ax.set_title(f"Prediksi Harga Saham Masa Depan dalam {n_future} {unit} ke Depan", fontsize=14)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga yang Diprediksi", fontsize=12)
    ax.grid(True)

    fig.autofmt_xdate()
    st.pyplot(fig)
    st.write(df_future)
# Menampilkan info apabila belum bisa memulai prediksi
else:
    valid_frequencies = ["Harian", "Mingguan", "Bulanan"]
    if not can_start_prediction and model_option == "Gunakan model dari database":
        show_extract_model_info_for_home_page()
    elif freq not in valid_frequencies:
        st.error(f"Frekuensi '{freq}' tidak dikenali. Pilih dari: {', '.join(valid_frequencies)}")
        st.stop()
    else:
        st.info("Silakan isi parameter di sidebar, lalu tekan 'Mulai Prediksi'.")
        show_extract_model_info_for_home_page()