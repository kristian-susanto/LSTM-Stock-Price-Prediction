import os
import io
import json
import time
import math
import base64
import tempfile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from io import BytesIO
from datetime import datetime
from pandas.tseries.offsets import BDay, Week, MonthEnd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
from utils.db import get_model_collection, get_model_metadata_collection

def list_model():
    """Mengembalikan daftar nama model yang tersimpan di database."""
    models_col = get_model_collection()
    return [doc["name"] for doc in models_col.find()]

def save_model_file(model_name, model, username, role):
    """Menyimpan model ke database dalam bentuk base64 dengan metadata pengguna."""
    models_col = get_model_collection()

    fd, path = tempfile.mkstemp(suffix=".keras")
    os.close(fd)

    try:
        model.save(path)
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
    finally:
        os.remove(path)

    models_col.update_one(
        {"name": model_name},
        {
            "$set": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": username,
                "role": role,
                "model_binary": encoded
            }
        },
        upsert=True
    )

def get_model_file(model_name):
    """Mengambil dokumen model mentah dari database berdasarkan nama."""
    models_col = get_model_collection()
    doc = models_col.find_one({"name": model_name})
    return doc if doc else {}

def load_model_file(model_name):
    """Memuat model dari database dan mengembalikannya sebagai objek Keras."""
    models_col = get_model_collection()
    doc = models_col.find_one({"name": model_name})
    if doc and "model_binary" in doc:
        decoded = base64.b64decode(doc["model_binary"])
        fd, path = tempfile.mkstemp(suffix=".keras")
        os.close(fd)
        try:
            with open(path, "wb") as f:
                f.write(decoded)
            return load_model(path)
        finally:
            os.remove(path)
    else:
        raise ValueError("Model tidak ditemukan atau file korup.")

def delete_model_file(model_name):
    """Menghapus model dari database berdasarkan nama."""
    models_col = get_model_collection()
    models_col.delete_one({"name": model_name})

def list_model_metadata():
    """Mengembalikan daftar metadata model yang tersimpan."""
    metadata_col = get_model_metadata_collection()
    return [{"name": doc["name"]} for doc in metadata_col.find()]

def save_model_metadata_file(metadata_name, combined_data_dict, username, role):
    """Menyimpan metadata model ke database."""
    metadata_col = get_model_metadata_collection()
    metadata_col.update_one(
        {"name": metadata_name},
        {
            "$set": {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": username,
                "role": role,
                "data": combined_data_dict
            }
        },
        upsert=True
    )

def get_model_metadata_file(metadata_name: str):
    """Mengambil metadata model mentah dari database berdasarkan nama."""
    metadata_col = get_model_metadata_collection()
    doc = metadata_col.find_one({"name": metadata_name})
    return doc if doc else {}

def load_model_metadata_file(metadata_name: str):
    """Memuat data metadata model dari database."""
    metadata_col = get_model_metadata_collection()
    doc = metadata_col.find_one({"name": metadata_name})
    return doc["data"] if doc and "data" in doc else None

def delete_model_metadata_file(metadata_name: str):
    """Menghapus metadata model dari database berdasarkan nama."""
    metadata_col = get_model_metadata_collection()
    metadata_col.delete_one({"name": metadata_name})

def parse_date(date_str):
    """Mengonversi string tanggal ke objek date dengan format yang dikenali."""
    for fmt in ("%Y/%m/%d", "%Y-%m-%d"):
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    raise ValueError("Format tanggal tidak dikenali. Gunakan YYYY/MM/DD atau YYYY-MM-DD.")

def get_frequency_code(freq):
    """Mengembalikan kode frekuensi berdasarkan input teks frekuensi."""
    return {"Harian": "1d", "Mingguan": "1wk", "Bulanan": "1mo"}[freq]

def get_time_step(freq):
    """Mengembalikan panjang time step berdasarkan frekuensi."""
    return {"Harian": 30, "Mingguan": 4, "Bulanan": 12}[freq]

def save_info_model(model, freq, ticker, start_date, end_date, model_type, history, metadata):
    """Menyimpan model dan metadata dengan informasi tambahan pengguna dan waktu."""
    model_name = f"{ticker}_{freq}_{start_date}_{end_date}_{model_type}"

    username = st.session_state.get("username") or "-"
    role = st.session_state.get("role") or "guest"

    try:
        save_model_file(model_name, model, username, role)

        combined_data = {
            "history": history.history if history else None,
            **metadata
        }
        
        save_model_metadata_file(model_name, combined_data, username, role)
        st.success(f"Model dan metadata {model_type} terbaru untuk '{model_name}' berhasil disimpan.")

    except Exception as e:
        st.error(f"Gagal menyimpan model atau metadata: {e}")

def delete_old_model(freq, ticker, start_date, end_date, model_type):
    """Menghapus model dan metadata lama berdasarkan parameter nama."""
    model_name = f"{ticker}_{freq}_{start_date}_{end_date}_{model_type}"
    
    try:
        delete_model_file(model_name)
        delete_model_metadata_file(model_name)
        return True
    except Exception as e:
        st.warning(f"Gagal menghapus model lama atau metadata '{model_name}': {e}")
        return False

def create_dataset(dataset, time_step=1):
    """Membuat dataset input (X) dan target (y) untuk pelatihan model."""
    X, y = [], []
    for i in range(len(dataset)-time_step):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs=50, batch_size=32, callbacks=None):
    """Membangun dan melatih model LSTM menggunakan dataset dan parameter tertentu."""
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    start_time = time.time()

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=callbacks)
    epochs_trained = history.epoch[-1] + 1
    duration = time.time() - start_time

    return model, history, duration, epochs, epochs_trained, batch_size

def highlight_rows(row, min_rmse):
    """Menentukan warna latar baris berdasarkan nilai RMSE."""
    if float(row['RMSE']) == min_rmse:
        return ['background-color: #c6f6d5' for _ in row]
    elif row['Tipe Model'] in ['Baseline', 'Baseline (dari Database)']:
        return ['background-color: #d3e5ff' for _ in row]
    else:
        return ['' for _ in row]

def get_future_trading_dates(start_date, n_steps, freq="Harian"):
    """Menghasilkan tanggal perdagangan masa depan berdasarkan frekuensi dan langkah waktu."""
    future_dates = []
    current_date = pd.to_datetime(start_date)

    if freq == "Harian":
        offset = BDay(1)
    elif freq == "Mingguan":
        offset = Week(weekday=4)
    elif freq == "Bulanan":
        offset = MonthEnd(1)
    else:
        raise ValueError("Frekuensi tidak dikenali. Gunakan: Harian, Mingguan, atau Bulanan.")

    for _ in range(n_steps):
        current_date += offset
        future_dates.append(current_date)

    return future_dates

def dataset_information_summary(df, dataset_name="Dataset", expanded=True):
    """Menampilkan ringkasan dan struktur informasi dataset di antarmuka Streamlit."""
    with st.expander(dataset_name, expanded=expanded):
        st.dataframe(df)
        st.write("Ringkasan informasi")
        non_null_counts = df.notnull().sum()
        info_table = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": [f"{count} non-null" for count in non_null_counts],
            "Dtype": df.dtypes.astype(str).values
        })
        st.write(info_table)

        buffer = io.StringIO()
        df.info(buf=buffer, verbose=False)
        full_info_str = buffer.getvalue()
        st.code(full_info_str, language="text")

def generate_lstm_model_config(model, time_step, epochs, epochs_trained, batch_size, title="Model Configuration"):
    """Menampilkan konfigurasi parameter model LSTM yang telah dilatih."""
    lstm_layers = [layer for layer in model.layers if isinstance(layer, LSTM)]
    dropout_layers = [layer for layer in model.layers if isinstance(layer, Dropout)]
    output_layer = model.layers[-1]

    config = {
        "Nama Metode": "LSTM",
        "Nama Model LSTM": type(model).__name__,
        "Time Step": time_step,
        "Jumlah Neuron per LSTM Layer": ", ".join(str(layer.units) for layer in lstm_layers),
        "Jumlah LSTM Layer": len(lstm_layers),
        "Learning Rate": round(float(model.optimizer.learning_rate.numpy()), 5),
        "Epochs": epochs,
        "Epochs Terlatih": epochs_trained,
        "Batch Size": batch_size,
        "Dropout Rate": ", ".join(str(layer.rate) for layer in dropout_layers) if dropout_layers else "0",
        "Optimizer": type(model.optimizer).__name__,
        "Loss": model.loss if isinstance(model.loss, str) else model.loss.__name__,
        "Output Layer": type(output_layer).__name__ + f"({output_layer.units})" if hasattr(output_layer, "units") else str(output_layer)
    }

    st.markdown(f"##### {title}")
    st.markdown(
        f"""
        <div style='text-align: justify; margin-bottom: 10px'>
            Model ini dilatih menggunakan parameter data yang sudah diproses dari mempelajari data historis.
        </div>
        """,
        unsafe_allow_html=True
    )
    output_text = ""
    for key, value in config.items():
        output_text += f"- **{key}**: {value}\n"

    st.markdown(output_text)

def model_architecture_summary(model):
    """Menampilkan ringkasan arsitektur model dan struktur layer dalam bentuk tabel."""
    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + "\n"))
    model_summary_str = summary_buffer.getvalue()
    st.code(model_summary_str, language="text")

    summary_lines = model_summary_str.strip().split("\n")
    table_start = next(i for i, line in enumerate(summary_lines) if "Layer (type)" in line)
    table_end = next(i for i, line in enumerate(summary_lines) if "Total params" in line)
    table_data = summary_lines[table_start + 2 : table_end - 1]

    parsed_rows = []
    for row in table_data:
        clean_row = row.replace("│", "").replace("─", "").replace("└", "").replace("┌", "").replace("┐", "").replace("┘", "")
        parts = [part.strip() for part in clean_row.strip().split("  ") if part.strip()]
        
        if len(parts) >= 3:
            layer_type = parts[0]
            output_shape = parts[1]
            param_count = parts[2]
            parsed_rows.append((layer_type, output_shape, param_count))
    
    df_model_summary = pd.DataFrame(parsed_rows, columns=["Layer (type)", "Output Shape", "Param #"])
    st.write(df_model_summary)

def show_prediction_results(y_true_rescaled, y_pred_rescaled, plot_dates, title="Hasil Prediksi", show_table=True):
    """Menampilkan hasil prediksi model dan evaluasi performa dalam grafik dan tabel."""
    df_result = pd.DataFrame({
        "Sebenarnya": y_true_rescaled.flatten(),
        "Prediksi": y_pred_rescaled.flatten(),
        "Galat": y_true_rescaled.flatten() - y_pred_rescaled.flatten(),
    })
    df_result["Galat (%)"] = np.where(
        df_result["Sebenarnya"] == 0, 0,
        df_result["Galat"] / df_result["Sebenarnya"] * 100
    )
    df_result["Tanggal"] = plot_dates.reset_index(drop=True)

    cols = ["Tanggal"] + [col for col in df_result.columns if col != "Tanggal"]
    df_result = df_result[cols]

    if show_table:
        st.write(df_result.head())

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_result["Tanggal"], df_result["Sebenarnya"], label="Sebenarnya")
    ax.plot(df_result["Tanggal"], df_result["Prediksi"], label="Prediksi")
    ax.set_title(title)
    ax.set_xlabel("Tahun")
    ax.set_ylabel("Harga")
    ax.legend()
    ax.grid(True)
    fig.autofmt_xdate()
    st.pyplot(fig)

    fig, ax = plt.subplots()
    sns.histplot(df_result["Galat"], bins=30, kde=True, ax=ax)
    ax.set_title("Distribusi Galat")
    ax.set_xlabel("Galat")
    ax.set_ylabel("Jumlah")
    st.pyplot(fig)

    rmse = math.sqrt(mean_squared_error(df_result["Sebenarnya"], df_result["Prediksi"]))
    mape = mean_absolute_percentage_error(df_result["Sebenarnya"], df_result["Prediksi"])

    return df_result, rmse, mape