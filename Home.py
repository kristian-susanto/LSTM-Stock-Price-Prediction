import os
import io
import time
import math
import platform
import datetime
import psutil
import matplotlib
import sklearn
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from utils.train_predict import get_frequency_code, get_time_step, create_dataset, build_and_train_model, show_prediction_results, get_future_trading_dates
from utils.model_utils import save_info_model, find_model_file, load_info_model, delete_old_model
from utils.viz_utils import dataset_information_summary, generate_lstm_model_config, model_architecture_summary, highlight_rows

st.set_page_config(page_title="Prediksi Harga Saham", page_icon="assets/favicon.ico", layout="wide")

st.header("Analisis Prediksi Harga Saham Menggunakan Metode LSTM", divider="gray")

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

st.sidebar.header("Pengaturan Model")
ticker = st.sidebar.text_input(
    "Masukkan Ticker Saham",
    "HLAG.DE",
    help="Ticker adalah kode unik yang digunakan untuk mewakili nama perusahaan pada saham."
)
start_date = st.sidebar.date_input("Tanggal Awal", datetime.date(1980, 1, 1))
end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())
freq = st.sidebar.selectbox(
    "Frekuensi Data",
    options=["Harian", "Mingguan", "Bulanan"],
    help="Frekuensi data merupakan frekuensi pengambilan data harga saham."
)
model_option = st.sidebar.radio(
    "Metode Pemodelan",
    ["Latih model baru", "Gunakan model dari database"],
    help="Pemilihan metode `Latih model baru` disarankan untuk prediksi terkini. Default ticker pada metode Gunakan model dari database adalah ticker `HLAG.DE`"
)
tune_model = st.sidebar.checkbox(
    "Aktifkan Model Tuning",
    value=True,
    help="Mengombinasikan parameter model tuning secara otomatis untuk analisis prediksi yang mendalam."
)

if st.sidebar.button("Mulai Prediksi"):
    st.subheader("0. System and Library Information")
    st.markdown("#### 0.1 Sistem")
    device_name = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
    gpus = tf.config.list_physical_devices('GPU')
    cpu_info = platform.processor()
    ram_info = psutil.virtual_memory()

    components = [
        "Sistem Operasi",
        "Versi Python",
        "Perangkat Komputasi",
        "Total RAM (GB)"
    ]
    info_values = [
        f"{platform.system()} {platform.release()}",
        platform.python_version(),
        device_name,
        round(ram_info.total / (1024 ** 3), 2)
    ]

    if cpu_info and cpu_info.strip():
        components.insert(3, "Informasi CPU")
        info_values.insert(3, cpu_info)

    if gpus:
        components.insert(3, "Jumlah GPU")
        info_values.insert(3, str(len(gpus)))
        components.insert(4, "Nama GPU")
        info_values.insert(4, ", ".join([gpu.name for gpu in gpus]))

    system_info = {
        "Komponen": components,
        "Informasi": info_values
    }
    st.dataframe(pd.DataFrame(system_info))

    st.markdown("#### 0.2 Library")
    env_info = {
        "Streamlit": st.__version__,
        "TensorFlow": tf.__version__,
        "NumPy": np.__version__,
        "Pandas": pd.__version__,
        "Sklearn": sklearn.__version__,
        "Matplotlib": matplotlib.__version__,
        "Seaborn": sns.__version__
    }

    env_df = pd.DataFrame(list(env_info.items()), columns=["Library", "Versi"])
    st.dataframe(env_df)

    st.subheader("1. Business Understanding")
    st.markdown(
        """
        <div style='text-align: justify'>
            Dalam dunia investasi dan pasar modal, kemampuan untuk memprediksi harga saham secara akurat 
            sangat penting bagi pengambilan keputusan yang tepat dan strategis. Oleh karena itu, dibutuhkan 
            metode yang mampu menangkap pola waktu (time series) secara efektif. Long Short-Term Memory (LSTM) 
            dirancang untuk mengenali pola dalam data berurutan dan memiliki keunggulan dalam mengatasi masalah 
            long-term dependencies. Dengan menerapkan LSTM, perusahaan atau investor dapat memperoleh prediksi 
            harga saham yang mendukung perencanaan dan manajemen risiko investasi.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("2. Data Understanding")
    with st.spinner("Mengambil data dari Yahoo Finance..."):
        if start_date >= end_date:
            st.sidebar.error("Tanggal awal harus lebih kecil dari tanggal akhir.")
            st.stop()

        freq_code = get_frequency_code(freq)
        data = yf.download(ticker, start=start_date, end=end_date + pd.Timedelta(days=1), interval=freq_code, auto_adjust=True)

        if data.empty:
            st.error(f"Data dengan ticker {ticker} tidak ditemukan atau gagal diunduh.")
        else:
            actual_start_date = data.index.min().to_pydatetime().date()
            actual_end_date = data.index.max().to_pydatetime().date()

            st.write(
                f"Data dengan ticker **{ticker}** dari tanggal "
                f"**{actual_start_date.strftime('%d %b, %Y')}** sampai "
                f"**{actual_end_date.strftime('%d %b, %Y')}** berhasil diunduh dari Yahoo Finance."
                f"Dataset dapat dilihat pada tautan [ini](https://finance.yahoo.com/quote/{ticker}/history)."
            )
    dataset_information_summary(data, "Data mentah", expanded=True)

    st.subheader("3. Data Preparation")
    st.markdown("#### 3.1 Data Transformation")
    with st.expander("Mengganti multiIndex dengan indeks biasa."):
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
            data.columns.name = None
        st.write(data.head())
    
    with st.expander("Mengubah indeks Date menjadi feature dan mengonversi tipe datanya ke datetime."):
        data.reset_index(inplace=True)
        data["Date"] = pd.to_datetime(data["Date"])
        st.write(data.head())
    
    st.markdown("#### 3.2 Data Filtering")
    st.write("Mengambil feature Date dan Close.")
    df = data[["Date", "Close"]].copy()
    df["Close"] = df["Close"].round(2)
    dataset_information_summary(df, "Data yang difilter", expanded=True)

    st.subheader("3.3 Data Cleaning")
    st.markdown("##### 3.3.1 Missing Values")
    with st.expander("Memeriksa dan menangani nilai yang hilang pada feature Close."):
        missing_before = df[["Close"]].isnull().sum()

        if df["Close"].isnull().sum() > 0:
            df["Close"] = df["Close"].fillna(method="ffill")

        missing_after = df[["Close"]].isnull().sum()
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
    st.write("Ringkasan statistik untuk feature Close.")
    st.write(df["Close"].describe())
    df["Year"] = df["Date"].dt.year
    year_counts = df["Year"].value_counts().sort_index()

    st.write("Distribusi jumlah data per tahun.")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_counts.index.astype(str), year_counts.values, color="skyblue")
    ax.set_xlabel("Tahun", fontsize=12)
    ax.set_ylabel("Jumlah Data", fontsize=12)
    ax.set_title("Distribusi Jumlah Data per Tahun", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig)

    st.write("Grafik harga saham.")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df["Date"], df["Close"], color="blue")
    ax.set_title("Grafik Harga Saham", fontsize=14)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga", fontsize=12)
    ax.grid(True)
    st.pyplot(fig)

    st.markdown("#### 3.5 Normalization")
    scaler = MinMaxScaler()
    df["Close Scaled"] = scaler.fit_transform(df[["Close"]])
    df_normalized = pd.DataFrame(df[["Date", "Close Scaled"]])
    df_normalized = df_normalized.rename(columns={"Date": "Tanggal", "Close Scaled": "Close yang Diskalasikan"})
    dataset_information_summary(df_normalized, "Data yang telah dinormalisasi tetapi belum disusun sebagai suatu sequence (urutan)", expanded=False)

    st.markdown("#### 3.6 Data Windowing")
    time_step = get_time_step(freq)
    data_scaled = df["Close Scaled"].values.reshape(-1, 1)

    X_seq, y_seq = create_dataset(data_scaled, time_step)

    df_seq = pd.DataFrame({
        "X (sequence)": [x.flatten().round(4).tolist() for x in X_seq],
        "y (target)": y_seq.flatten().round(4)
    })
    dataset_information_summary(df_seq, "Data yang telah dinormalisasi dan disusun sebagai sequence untuk input (X) dan target (y)", expanded=False)

    st.markdown("#### 3.7 Data Splitting")
    time_step = get_time_step(freq)
    X, y = create_dataset(data_scaled, time_step)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    full_data = np.concatenate([y_train, y_test])
    full_data_rescaled = scaler.inverse_transform(full_data.reshape(-1, 1))

    plot_dates = df["Date"].iloc[time_step:].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(plot_dates, full_data_rescaled, label="Harga", color="blue")

    split_date = plot_dates.iloc[split]
    ax.axvline(split_date, color="red", linestyle="--", label="Train/Test Split")

    ax.fill_between(plot_dates[:split], full_data_rescaled[:split, 0], color="lightblue", alpha=0.5, label="Train")
    ax.fill_between(plot_dates[split:], full_data_rescaled[split:, 0], color="orange", alpha=0.5, label="Test")

    ax.set_title("Pembagian Data Train dan Test")
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    df_X_and_y_train = pd.DataFrame({
        "X_train (sequence)": X_train[i].flatten().round(4).tolist(),
        "y_train": round(float(y_train[i]), 4)
    } for i in range(len(X_train)))
    dataset_information_summary(df_X_and_y_train, "Kelompok data X_train dan y_train", expanded=False)

    df_X_and_y_test = pd.DataFrame({
        "X_test (sequence)": X_test[i].flatten().round(4).tolist(),
        "y_test": round(float(y_test[i]), 4)
    } for i in range(len(X_test)))
    dataset_information_summary(df_X_and_y_test, "Kelompok data X_test dan y_test", expanded=False)

    st.subheader("4. Modeling")
    if model_option == "Gunakan model dari database":
        model_path = find_model_file(freq, ticker, model_type="baseline")
        if model_path:
            st.success("Model baseline telah ditemukan.")
            params = load_info_model(freq=freq, info_type="metadata", ticker=ticker, model_type="baseline")
            if params is None:
                st.error("Metadata model baseline tidak ditemukan.")
                st.stop()
            else:
                time_step = params["time_step"]
                epochs = params["epochs"]
                batch_size = params["batch_size"]
                rmse = params["rmse"]
                mape = params["mape"]
                duration = params["duration"]

            X, y = create_dataset(data_scaled, time_step)
            split = int(len(X) * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            model = load_info_model(info_type="model", ticker=ticker, model_path=model_path)
            start_time = time.time()
            y_pred = model.predict(X_test)
            duration = time.time() - start_time
            history = None

            time_step = time_step
            epochs = epochs
            batch_size = batch_size
        else:
            st.error("Model baseline tidak ditemukan.")
            st.stop()
    elif model_option == "Latih model baru":
        with st.spinner("Melatih model dan melakukan prediksi..."):
            model, history, duration, epochs, batch_size = build_and_train_model(X_train, y_train, X_test, y_test, time_step)

    st.markdown("#### 4.1 Model Selection")
    generate_lstm_model_config(
        model=model,
        time_step=time_step,
        epochs=epochs,
        batch_size=batch_size,
        title="4.1.1 Model Configuration"
    )

    st.markdown("##### 4.1.2 Model Architecture Summary")
    model_architecture_summary(model)

    st.markdown("#### 4.2 Model Training")
    fig, ax = plt.subplots(figsize=(8, 4))

    if history is None:
        model = load_info_model(info_type="model", ticker=ticker, model_path=model_path)
        history_dict = load_info_model(info_type="history", ticker=ticker, model_path=model_path)
        history = type("History", (), {"history": history_dict}) if history_dict else None

    if history is not None:
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

    y_pred = model.predict(X_test)
    y_pred_rescaled = scaler.inverse_transform(y_pred)
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    st.markdown("#### 4.3 Model Inference")
    st.markdown("##### 4.3.1 Prediction Results")
    plot_dates_test = df["Date"].iloc[time_step + len(y_train):]

    df_result, rmse, mape = show_prediction_results(
        y_true_rescaled=y_test_rescaled,
        y_pred_rescaled=y_pred_rescaled,
        plot_dates=plot_dates_test,
        title="Hasil Prediksi"
    )

    baseline_tipe = "Baseline (dari Database)" if model_option == "Gunakan model dari database" else "Baseline"

    baseline_results = [{
        "time_step": time_step,
        "epochs": epochs,
        "batch_size": batch_size,
        "rmse": rmse,
        "mape": mape,
        "model": model,
        "history": history,
        "duration": duration,
        "tipe": baseline_tipe
    }]

    if model_option == "Latih model baru":
        if st.session_state.username == "admin":
            metadata = {
                "time_step": time_step,
                "epochs": epochs,
                "batch_size": batch_size,
                "rmse": rmse,
                "mape": mape,
                "duration": duration,
            }

            files_deleted = delete_old_model(freq, ticker, model_type="baseline")
            save_info_model(model, freq, ticker=ticker, model_type="baseline", history=history, metadata=metadata)

            if files_deleted:
                st.success(f"Model baseline lama berhasil dihapus. Model, histori training, dan metadata parameter baseline yang baru telah berhasil disimpan.")
            else:
                st.success(f"Model, histori training, dan metadata parameter baseline telah berhasil disimpan.")

    if tune_model:
        st.markdown("##### 4.3.2 Model Results")
        st.write("Hasil model baseline.")
    else:
        st.subheader("5. Evaluation")
        st.markdown("#### 5.1 Model Evaluation")
        st.write("Hasil evaluasi model.")

    baseline_tipe = "Baseline (dari Database)" if model_option == "Gunakan model dari database" else "Baseline"

    baseline_table = pd.DataFrame([{
        "Tipe Model": baseline_tipe,
        "Time Step": time_step,
        "Epoch": epochs,
        "Batch Size": batch_size,
        "RMSE": round(rmse, 4),
        "MAPE (%)": round(mape * 100, 2),
        "Durasi (detik)": round(duration, 2)
    }])
    baseline_table = (
        baseline_table
        .style
        .format({
            "RMSE": "{:.4f}",
            "MAPE (%)": "{:.2f}",
            "Durasi (detik)": "{:.2f}"
        })
    )
    st.dataframe(baseline_table)

    if tune_model:
        st.markdown("#### 4.4 Model Tuning")
        if model_option == "Gunakan model dari database":
            model_path = find_model_file(freq, ticker, model_type="best tuning")
            if model_path:
                st.success("Model best tuning telah ditemukan.")
                params = load_info_model(freq=freq, info_type="metadata", ticker=ticker, model_type="best tuning")
                if params is None:
                    st.error("Metadata model best tuning tidak ditemukan.")
                    st.stop()
                else:
                    best_model = load_info_model(info_type="model", ticker=ticker, model_path=model_path)
                    time_step = params["time_step"]
                    epochs = params["epochs"]
                    batch_size = params["batch_size"]
                    rmse = params["rmse"]
                    mape = params["mape"]
                    duration = params["duration"]

                    best = {
                        "model": best_model,
                        "history": None,
                        "time_step": time_step,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "rmse": rmse,
                        "mape": mape,
                        "duration": duration,
                        "tipe": "Tuning (dari Database)"
                    }
                st.markdown("##### 4.5.1 Best Model Tuning (from Database)")
                X, y = create_dataset(data_scaled, time_step)
                split = int(len(X)*0.8)
                X_train_tune, X_test_tune = X[:split], X[split:]
                y_train_tune, y_test_tune = y[:split], y[split:]

                X_train_tune = X_train_tune.reshape(X_train_tune.shape[0], X_train_tune.shape[1], 1)
                X_test_tune = X_test_tune.reshape(X_test_tune.shape[0], X_test_tune.shape[1], 1)

                best_model = load_info_model(info_type="model", ticker=ticker, model_path=model_path)
                start_time = time.time()
                y_pred = best_model.predict(X_test_tune)
                duration = time.time() - start_time
                y_pred_rescaled = scaler.inverse_transform(y_pred)
                y_test_rescaled = scaler.inverse_transform(y_test_tune.reshape(-1, 1))
                plot_dates_best = df["Date"].iloc[time_step + len(y_train_tune):]

                df_tuning_result = pd.DataFrame({
                    "Date": plot_dates_best.reset_index(drop=True),
                    "Actual": y_test_rescaled.flatten(),
                    "Predicted": y_pred_rescaled.flatten()
                })

                rmse_best = math.sqrt(mean_squared_error(df_tuning_result["Actual"], df_tuning_result["Predicted"]))
                mape_best = mean_absolute_percentage_error(df_tuning_result["Actual"], df_tuning_result["Predicted"])

                tuning_results = []
                
                tuning_results.append({
                    "time_step": time_step,
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "rmse": rmse_best,
                    "mape": mape_best,
                    "model": best_model,
                    "history": None,
                    "duration": duration,
                    "tipe": "Tuning (dari Database)"
                })
                best = tuning_results[0]

            else:
                st.error("Model best tuning tidak ditemukan.")
                st.stop()
        else:
            st.markdown("##### 4.4.1 Hyperparameter Tuning")

            if freq == "Harian":
                time_steps = [30, 60, 90]
            elif freq == "Mingguan":
                time_steps = [4, 8, 12]
            else:
                time_steps = [12, 24, 36]

            epochs_list = [50, 75, 100]
            batch_sizes = [16, 32]

            total_iter = len(time_steps) * len(epochs_list) * len(batch_sizes)
            progress_bar = st.progress(0)
            status_text = st.empty()
            tuning_results = []

            iter_count = 0
            for ts in time_steps:
                for ep in epochs_list:
                    for bs in batch_sizes:
                        iter_count += 1
                        status_text.text(f"Melatih model dengan time step = {ts}, epoch = {ep}, dan batch size = {bs} ({iter_count}/{total_iter})")

                        X, y = create_dataset(data_scaled, ts)
                        split = int(len(X)*0.8)
                        X_train_tune, X_test_tune = X[:split], X[split:]
                        y_train_tune, y_test_tune = y[:split], y[split:]

                        X_train_tune = X_train_tune.reshape(X_train_tune.shape[0], X_train_tune.shape[1], 1)
                        X_test_tune = X_test_tune.reshape(X_test_tune.shape[0], X_test_tune.shape[1], 1)

                        model_temp, history_temp, duration_temp, _, _ = build_and_train_model(X_train_tune, y_train_tune, X_test_tune, y_test_tune, ts, epochs=ep, batch_size=bs)
                        pred = model_temp.predict(X_test_tune)
                        pred = scaler.inverse_transform(pred)
                        actual = scaler.inverse_transform(y_test_tune.reshape(-1, 1))
                        rmse = math.sqrt(mean_squared_error(actual, pred))
                        mape = mean_absolute_percentage_error(actual, pred)

                        tuning_results.append({
                            "time_step": ts,
                            "epochs": ep,
                            "batch_size": bs,
                            "rmse": rmse,
                            "mape": mape,
                            "model": model_temp,
                            "history": history_temp,
                            "duration": duration_temp,
                            "tipe": "Tuning"
                        })

                        progress_bar.progress(iter_count / total_iter)

        best = sorted(tuning_results, key=lambda x: x["rmse"])[0]

        if st.session_state.username == "admin":
            metadata_best = {
                "time_step": best["time_step"],
                "epochs": best["epochs"],
                "batch_size": best["batch_size"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "duration": best["duration"],
            }
            files_deleted = delete_old_model(freq, ticker, model_type="best tuning")
            save_info_model(best["model"], freq, ticker=ticker, model_type="best tuning", history=best["history"], metadata=metadata_best)

            if files_deleted:
                st.success(f"Model best tuning lama berhasil dihapus. Model, histori training, dan metadata parameter best tuning yang baru telah berhasil disimpan.")
            else:
                st.success(f"Model, histori training, dan metadata parameter best tuning telah berhasil disimpan.")

        st.success(f"Model terbaik menggunakan parameter time step = {best['time_step']}, epoch = {best['epochs']}, dan batch size = {best['batch_size']} dengan metrik evaluasi RMSE sebesar {best['rmse']:.4f} dan MAPE sebesar {best['mape']:.2%}")

        st.markdown("#### 4.5 Best Model Selection")
        best_model = best["model"]
        best_history = best["history"]
        best_time_step = best["time_step"]
        best_epochs = best["epochs"]
        best_batch_size = best["batch_size"]
        best_duration = best["duration"]

        generate_lstm_model_config(
            model=best["model"],
            time_step=best["time_step"],
            epochs=best["epochs"],
            batch_size=best["batch_size"],
            duration=best["duration"],
            title="4.5.1 Best Model Configuration"
        )

        st.markdown("##### 4.5.2 Best Model Architecture Summary")
        model_architecture_summary(best_model)

        st.markdown("#### 4.6 Best Model Training")
        fig, ax = plt.subplots(figsize=(8, 4))

        if best["history"] is None:
            best_model = load_info_model(info_type="model", ticker=ticker, model_path=model_path)
            history_dict = load_info_model(info_type="history", ticker=ticker, model_path=model_path)
            best["history"] = type("History", (), {"history": history_dict}) if history_dict else None

        if best["history"] is not None:
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

        st.markdown("#### 4.7 Best Model Inference")
        st.markdown("##### 4.7.1 Best Model Prediction Results")
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

        st.markdown("##### 4.7.2 Best Model Results")
        st.write("Hasil model tuning.")
        best_tuning_table = pd.DataFrame([{
            "Tipe Model": "Tuning terbaik",
            "Time Step": best["time_step"],
            "Epoch": best["epochs"],
            "Batch Size": best["batch_size"],
            "RMSE": round(rmse_best, 4),
            "MAPE (%)": round(mape_best * 100, 2),
            "Durasi (detik)": round(best['duration'], 2)
        }])
        best_tuning_table = (
            best_tuning_table
            .style
            .format({
                "RMSE": "{:.4f}",
                "MAPE (%)": "{:.2f}",
                "Durasi (detik)": "{:.2f}"
            })
        )
        st.dataframe(best_tuning_table)

        st.subheader("5. Evaluation")
        st.markdown("#### 5.1 Final Model Evaluation")
        st.write("Hasil evaluasi model terbaik berdasarkan RMSE terendah.")

        all_results = baseline_results + tuning_results
        model_table = pd.DataFrame([{
            "Tipe Model": r["tipe"],
            "Time Step": r["time_step"],
            "Epoch": r["epochs"],
            "Batch Size": r["batch_size"],
            "RMSE": round(r["rmse"], 4),
            "MAPE (%)": round(r['mape'] * 100, 2),
            "Durasi (detik)": round(r['duration'], 2)
        } for r in all_results])

        model_table = model_table.sort_values(by="RMSE").reset_index(drop=True)

        min_rmse_value = model_table["RMSE"].min()
        model_table = (
            model_table
            .style
            .apply(lambda row: highlight_rows(row, min_rmse_value), axis=1)
            .format({
                "RMSE": "{:.4f}",
                "MAPE (%)": "{:.2f}",
                "Durasi (detik)": "{:.2f}"
            })
        )
        st.dataframe(model_table, use_container_width=True)

    st.markdown("#### 5.2 Forecasting")
    unit = {"Harian": "Hari", "Mingguan": "Minggu", "Bulanan": "Bulan"}[freq]
    st.info(
        "Forecasting ini bersifat estimasi dan tidak merepresentasikan kejadian nyata secara pasti. "
        "Model menghasilkan proyeksi berdasarkan pola historis dan bukan kejadian masa depan yang pasti.\n\n"
        f"Jumlah waktu sebesar **10 {unit.lower()} ke depan** dipilih karena merupakan angka psikologis "
        "umum yang digunakan untuk memberikan gambaran ringkas. Angka ini cukup untuk menunjukkan tren "
        "yang mungkin terjadi."
    )
    n_future = 10

    future_model = best["model"] if tune_model else model
    future_time_step = best["time_step"] if tune_model else time_step

    last_input = data_scaled[-future_time_step:]
    temp_input = list(last_input.flatten())

    st.write(f"Model yang digunakan adalah {'hasil tuning' if tune_model else 'baseline'} untuk memprediksi {n_future} {unit.lower()} ke depan.")

    progress_bar = st.progress(0)
    status_text = st.empty()
    predictions = []

    for i in range(n_future):
        status_text.text(f"Memproses iterasi ke-{i+1} dari {n_future}")
        x_input = np.array(temp_input[-future_time_step:]).reshape(1, future_time_step, 1)
        yhat = future_model.predict(x_input, verbose=0)
        temp_input.append(yhat[0][0])
        predictions.append(yhat[0][0])
        progress_bar.progress((i + 1) / n_future)

    start_date = df["Date"].max()
    future_dates = get_future_trading_dates(start_date, n_future, freq=freq)
    future_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))

    df_future = pd.DataFrame({"Tanggal": future_dates, "Harga yang Diprediksi": future_prices.flatten()})

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df_future["Tanggal"], df_future["Harga yang Diprediksi"], marker="o", linestyle="-")    
    ax.set_title(f"Prediksi Harga Saham Masa Depan dalam {n_future} {unit} ke Depan", fontsize=14)
    ax.set_xlabel("Tanggal", fontsize=12)
    ax.set_ylabel("Harga yang Diprediksi", fontsize=12)
    ax.grid(True)

    fig.autofmt_xdate()
    st.pyplot(fig)
    st.write(df_future)
else:
    st.info("Silakan isi parameter di sidebar, lalu tekan 'Mulai Prediksi'")