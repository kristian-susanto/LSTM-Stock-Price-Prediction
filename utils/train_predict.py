import time
import math
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import BDay, Week, MonthEnd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras import Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

def get_frequency_code(freq):
    return {"Harian": "1d", "Mingguan": "1wk", "Bulanan": "1mo"}[freq]

def get_time_step(freq):
    return {"Harian": 30, "Mingguan": 4, "Bulanan": 12}[freq]

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset)-time_step):
        X.append(dataset[i:(i+time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

def build_and_train_model(X_train, y_train, X_test, y_test, time_step, epochs=50, batch_size=32):
    model = Sequential()
    model.add(Input(shape=(time_step, 1)))
    model.add(LSTM(50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")

    start_time = time.time()

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size, verbose=1, callbacks=[early_stop])
    duration = time.time() - start_time

    return model, history, duration, epochs, batch_size

def show_prediction_results(
    y_true_rescaled,
    y_pred_rescaled,
    plot_dates,
    title="Hasil Prediksi",
    show_table=True
):
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
    ax.set_xlabel("Tanggal")
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

def get_future_trading_dates(start_date, n_steps, freq="Harian"):
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