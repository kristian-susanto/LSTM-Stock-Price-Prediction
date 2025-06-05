# Stock Price Prediction Analysis Using LSTM Method

## Description
This project aims to predict stock prices using the Long Short-Term Memory (LSTM) model as one of the Recurrent Neural Network (RNN) architectures that excels in handling time series data. This application is built using Python and can be deployed using Streamlit for easy access and visualization.

## Features
- Dataset retrieval using Yahoo Finance
- Visualization of historical stock price data
- Data pre-processing (normalization, time series window formation)
- LSTM model training with Keras and TensorFlow
- Model evaluation using RMSE and MAPE metrics
- Stock price forecasting
- Interactive interface using Streamlit

## Folder Structure
```
LSTM-Stock-Price-Prediction/
│
├── assets/
│   └── favicon.ico
│
├── datas/
│   ├── histories/
│   ├── metadatas/
│   ├── models/
│   └── users/
│
├── pages/
│   ├── 01_Login.py
│   ├── 02_Register.py
│   └── 03_Admin Panel.py
│
├── utils/
│   ├── auth.py
│   ├── model_utils.py
│   ├── train_predict.py
│   └── viz_utils.py
│
├── .gitignore
├── Home.py
├── README.MD
└── requirements.txt
```

## How to Run
### 1. Clone Repository
```bash
git clone https://github.com/kristian-susanto/LSTM-Stock-Price-Prediction.git
cd LSTM-Stock-Price-Prediction
```

### 2. Dependency Installation
It is recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows
```

Once the virtual environment is active, install all dependencies in one of the following ways:
Using the requirements.txt file:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install streamlit pandas numpy matplotlib seaborn yfinance scikit-learn psutil tensorflow bcrypt
```

### 3. Run Streamlit Application
```bash
streamlit run Home.py
```
The application will open in the browser automatically at the address `http://localhost:8501`.

## Contributor
- [Kristian Susanto](https://github.com/kristian-susanto)