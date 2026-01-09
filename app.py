# ==========================================
# Air Passengers Time Series Forecasting App
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# ------------------------------------------
# Page Config
# ------------------------------------------
st.set_page_config(
    page_title="Air Passengers Forecasting",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ Air Passengers Time Series Forecasting")
st.write("Forecast future airline passenger demand using SARIMA model")

# ------------------------------------------
# Load Data Function
# ------------------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("AirPassengers.csv")
    data["Month"] = pd.to_datetime(data["Month"])
    data.set_index("Month", inplace=True)
    return data

# ------------------------------------------
# ADF Test Function
# ------------------------------------------
def adf_test(series):
    result = adfuller(series)
    return result[0], result[1]

# ------------------------------------------
# Train SARIMA Model
# ------------------------------------------
@st.cache_resource
def train_model(log_series):
    model = SARIMAX(
        log_series,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit()
    return results

# ------------------------------------------
# Load Dataset
# ------------------------------------------
data = load_data()

# ------------------------------------------
# Sidebar Controls
# ------------------------------------------
st.sidebar.header("⚙️ Forecast Settings")
forecast_steps = st.sidebar.slider(
    "Forecast Months",
    min_value=12,
    max_value=48,
    value=24,
    step=6
)

# ------------------------------------------
# Dataset Preview
# ------------------------------------------
st.subheader("📄 Dataset Preview")
st.dataframe(data.head())

# ------------------------------------------
# Original Time Series Plot
# ------------------------------------------
st.subheader("📈 Passenger Trend")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(data["Passengers"], label="Passengers")
ax1.set_xlabel("Year")
ax1.set_ylabel("Passengers")
ax1.legend()
st.pyplot(fig1)

# ------------------------------------------
# Stationarity Check
# ------------------------------------------
st.subheader("🧪 Stationarity Check (ADF Test)")

adf_stat, p_value = adf_test(data["Passengers"])

st.write(f"**ADF Statistic:** {adf_stat:.4f}")
st.write(f"**p-value:** {p_value:.4f}")

if p_value <= 0.05:
    st.success("Series is stationary")
else:
    st.warning("Series is NOT stationary (Transformation required)")

# ------------------------------------------
# Log Transformation
# ------------------------------------------
data["Log_Passengers"] = np.log(data["Passengers"])

# ------------------------------------------
# Train Model
# ------------------------------------------
with st.spinner("Training SARIMA model..."):
    model_results = train_model(data["Log_Passengers"])

st.success("Model trained successfully")

# ------------------------------------------
# Forecast
# ------------------------------------------
forecast = model_results.get_forecast(steps=forecast_steps)
forecast_mean = np.exp(forecast.predicted_mean)
forecast_ci = forecast.conf_int()
forecast_ci = np.exp(forecast_ci)

# ------------------------------------------
# Forecast Plot
# ------------------------------------------
st.subheader("🔮 Forecast Result")

fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data["Passengers"], label="Actual")
ax2.plot(forecast_mean, label="Forecast", color="red")
ax2.fill_between(
    forecast_ci.index,
    forecast_ci.iloc[:, 0],
    forecast_ci.iloc[:, 1],
    color="pink",
    alpha=0.3,
    label="Confidence Interval"
)
ax2.set_xlabel("Year")
ax2.set_ylabel("Passengers")
ax2.legend()
st.pyplot(fig2)

# ------------------------------------------
# Model Evaluation
# ------------------------------------------
st.subheader("📊 Model Evaluation")

actual = data["Passengers"][-forecast_steps:]
predicted = forecast_mean[:len(actual)]

mae = mean_absolute_error(actual, predicted)

st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.2f}")

# ------------------------------------------
# Model Summary
# ------------------------------------------
with st.expander("📑 SARIMA Model Summary"):
    st.text(model_results.summary())

# ------------------------------------------
# Footer
# ------------------------------------------
st.markdown("---")
st.caption("Built with Python, SARIMA & Streamlit")
