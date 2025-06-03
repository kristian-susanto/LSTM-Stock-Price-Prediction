import io
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.layers import LSTM, Dropout

def dataset_information_summary(df, dataset_name="Dataset", expanded=True):
    with st.expander(dataset_name, expanded=expanded):
        st.dataframe(df)
        st.write("Ringkasan informasi")
        non_null_counts = df.notnull().sum()
        info_table = pd.DataFrame({
            "Column": df.columns,
            "Non-Null Count": [f"{count} non-null" for count in non_null_counts],
            "Dtype": df.dtypes.astype(str).values
        })
        st.dataframe(info_table)

        buffer = io.StringIO()
        df.info(buf=buffer, verbose=False)
        full_info_str = buffer.getvalue()
        st.code(full_info_str, language="text")

def generate_lstm_model_config(
    model, 
    time_step, 
    epochs, 
    batch_size, 
    duration=None, 
    title="Model Configuration"
):
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
        "Batch Size": batch_size,
        "Dropout Rate": ", ".join(str(layer.rate) for layer in dropout_layers) if dropout_layers else "0",
        "Optimizer": type(model.optimizer).__name__,
        "Loss": model.loss if isinstance(model.loss, str) else model.loss.__name__,
        "Output Layer": type(output_layer).__name__ + f"({output_layer.units})" if hasattr(output_layer, "units") else str(output_layer)
    }

    if duration is not None:
        config["Training Duration (s)"] = round(duration, 2)

    df = pd.DataFrame(list(config.items()), columns=["Parameter", "Nilai"])
    st.markdown(f"##### {title}")
    st.dataframe(df)

def model_architecture_summary(model):
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
    st.dataframe(df_model_summary, use_container_width=True)

def highlight_rows(row, min_rmse):
    if float(row['RMSE']) == min_rmse:
        return ['background-color: #c6f6d5' for _ in row]
    elif row['Tipe Model'] in ['Baseline', 'Baseline (dari Database)']:
        return ['background-color: #d3e5ff' for _ in row]
    else:
        return ['' for _ in row]