
# day49_nbeats.py

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

st.title("📈 Day 49 - N-BEATS Time Series Forecasting")

# --- Generate synthetic time series data ---
np.random.seed(42)
time = np.arange(0, 200, 0.1)
series = np.sin(0.2 * time) + np.sin(0.05 * time) + np.random.normal(scale=0.3, size=len(time))

# Scale data
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()

# Sidebar controls
window_size = st.sidebar.slider("Window Size", 10, 50, 30)
epochs = st.sidebar.slider("Epochs", 5, 50, 20)
batch_size = st.sidebar.slider("Batch Size", 16, 128, 32)

# --- Prepare dataset ---
def create_dataset(series, window_size):
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i:i+window_size])
        y.append(series[i+window_size])
    return np.array(X), np.array(y)

X, y = create_dataset(series_scaled, window_size)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# --- N-BEATS-like model ---
def build_nbeats(input_dim):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for _ in range(4):
        x = layers.Dense(256, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs, name="N-BEATS")
    return model

model = build_nbeats(window_size)
model.compile(optimizer="adam", loss="mse")

st.write("### Training Model...")
with st.spinner("Training in progress..."):
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=epochs, batch_size=batch_size, verbose=0)

# --- Forecast ---
y_pred = model.predict(X_test)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()
y_pred_inv = scaler.inverse_transform(y_pred).flatten()

# --- Plot Results ---
fig, ax = plt.subplots(figsize=(10,6))
ax.plot(range(len(y_test_inv)), y_test_inv, label="Actual")
ax.plot(range(len(y_pred_inv)), y_pred_inv, label="Predicted")
ax.set_title("N-BEATS Time Series Forecasting")
ax.set_xlabel("Time Steps")
ax.set_ylabel("Value")
ax.legend()
st.pyplot(fig)

# --- Show Loss Curve ---
fig2, ax2 = plt.subplots()
ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Training Loss Curve")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("MSE Loss")
ax2.legend()
st.pyplot(fig2)

st.success("✅ Forecasting complete!")


