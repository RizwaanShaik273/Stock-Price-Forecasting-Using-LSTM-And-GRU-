import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
import streamlit as st
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

# Fetch stock data
def get_stock_data(ticker, start='2015-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end)
    return df[['Close']]

# Prepare data for LSTM/GRU
def prepare_data(data, time_step=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data)
    X, y = [], []
    for i in range(time_step, len(data_scaled)):
        X.append(data_scaled[i-time_step:i, 0])
        y.append(data_scaled[i, 0])
    return np.array(X), np.array(y), scaler, data_scaled

# Build LSTM Model
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Build GRU Model
def build_gru(input_shape):
    model = Sequential([
        GRU(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        GRU(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Predict Future Prices based on Date
def predict_until_date(model, last_sequence, scaler, future_date, last_known_date):
    future_predictions = []
    input_seq = last_sequence.copy()
    days = (future_date - last_known_date).days
    
    for _ in range(days):
        pred = model.predict(input_seq.reshape(1, len(input_seq), 1))
        future_predictions.append(pred[0, 0])
        input_seq = np.append(input_seq[1:], pred[0, 0])
    
    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# Streamlit App
st.title("Stock Price Forecasting Using LSTM & GRU")

# User Input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", "AAPL")
future_date = st.date_input("Select Future Date:", datetime.today() + timedelta(days=30))

if st.button("Predict"):
    df = get_stock_data(ticker)
    if df.empty:
        st.error("No data available for the selected stock ticker.")
    else:
        time_step = 60
        X, y, scaler, data_scaled = prepare_data(df.values, time_step)
        X = X.reshape((X.shape[0], X.shape[1], 1))
        train_size = int(len(X) * 0.8)
        X_train, X_test, y_train, y_test = X[:train_size], X[train_size:], y[:train_size], y[train_size:]
        
        # Get the last sequence from the entire scaled data
        last_sequence = data_scaled[-time_step:, 0]  # Corrected line
        
        # Train LSTM Model
        lstm_model = build_lstm((X_train.shape[1], 1))
        lstm_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Train GRU Model
        gru_model = build_gru((X_train.shape[1], 1))
        gru_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
        
        # Predict Future Prices
        last_known_date = df.index[-1].to_pydatetime().date()
        if future_date <= last_known_date:
            st.error("Future date must be after the last known trading date.")
        else:
            # Predict using the corrected last_sequence
            future_lstm = predict_until_date(lstm_model, last_sequence, scaler, future_date, last_known_date)
            future_gru = predict_until_date(gru_model, last_sequence, scaler, future_date, last_known_date)
            
            # Display Predictions
            st.subheader(f"Predicted Price for {future_date.strftime('%Y-%m-%d')}")
            st.write(f"LSTM Prediction: ${future_lstm[-1][0]:.2f}")
            st.write(f"GRU Prediction: ${future_gru[-1][0]:.2f}")