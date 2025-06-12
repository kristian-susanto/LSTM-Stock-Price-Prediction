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
├── pages/
│   ├── 01_Login.py
│   ├── 02_Register.py
│   └── 03_Dashboard.py
│
├── utils/
│   ├── auth.py
│   ├── db.py
│   └── model.py
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
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow yfinance bcrypt pymongo python-dotenv streamlit
```

### 3. .env File Configuration
Create a file named .env in the project root directory, then add your MongoDB configuration as follows:
```bash
MONGO_URI="mongodb+srv://<username>:<password>@<cluster>.mongodb.net/my_app_db?retryWrites=true&w=majority"
MONGO_DB="your_database_name"
```
Replace <username>, <password>, and <cluster> with your MongoDB credentials and database details.
If you are using the MongoDB Atlas service, make sure your IP address has been whitelisted and that access from external applications has been enabled.

### 4. Run Streamlit Application
```bash
streamlit run Home.py
```
The application will open in the browser automatically at the address `http://localhost:8501`.

## Contributor
- [Kristian Susanto](https://github.com/kristian-susanto)
