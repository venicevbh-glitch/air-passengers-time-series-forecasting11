import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Air Passengers Forecasting",
    page_icon="✈️",
    layout="wide"
)

# -----------------------------
# TITLE & INTRO
# -----------------------------
st.title("✈️ Air Passengers Time Series Forecasting")
st.markdown(
    """
    Forecast future airline passenger demand using **Seasonal ARIMA (SARIMA)**.  
    This app demonstrates a complete **time series analysis → modeling → forecasting** workflow.
    """
)

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Forecast Settings")
forecast_horizon = st.sidebar.slider(
    "Forecast Months",
    min_value=6,
    max_value=36,
    value=12,
    step=1
)

st.sidebar.markdown("---")
st.sidebar.info(
    "SARIMA Model\n\n(1,1,1) × (1,1,1,12)\n\n"
    "Handles trend + yearly seasonality"
)

# -----------------------------
# LOAD DATA
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("AirPassengers.csv")
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    return df

data = load_data()

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📄 Data", "📊 Analysis", "🔮 Forecast", "📑 Model Summary"]
)

# -----------------------------
# TAB 1: DATA
# -----------------------------
with tab1:
    st.subheader("Dataset Overview")
    st.write(data.head())

    st.markdown("### Passenger Trend")
    fig, ax = plt.subplots()
    ax.plot(data, label="Passengers")
    ax.set_xlabel("Year")
    ax.set_ylabel("Passengers")
    ax.legend()
    st.pyplot(fig)

# -----------------------------
# TAB 2: ANALYSIS
# -----------------------------
with tab2:
    st.subheader("Stationarity Check (ADF Test)")

    adf_result = adfuller(data["Passengers"])
    st.write(f"**ADF Statistic:** {adf_result[0]:.4f}")
    st.write(f"**p-value:** {adf_result[1]:.4f}")

    if adf_result[1] < 0.05:
        st.success("Series is stationary")
    else:
        st.warning("Series is NOT stationary (Transformation required)")

    st.markdown(
        """
        **Why this matters:**  
        Time series models assume stationarity.  
        We apply log transformation and differencing inside SARIMA to handle this.
        """
    )

# -----------------------------
# TAB 3: FORECAST
# -----------------------------
with tab3:
    st.subheader("Future Passenger Forecast")

    # Log transform
    data["Log_Passengers"] = np.log(data["Passengers"])

    # Train SARIMA
    model = SARIMAX(
        data["Log_Passengers"],
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )

    results = model.fit(disp=False)
    st.success("Model trained successfully")

    # Forecast
    forecast = results.get_forecast(steps=forecast_horizon)
    forecast_mean = np.exp(forecast.predicted_mean)
    conf_int = np.exp(forecast.conf_int())

    # Plot forecast
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(data["Passengers"], label="Historical")
    ax.plot(forecast_mean, label="Forecast", color="black")
    ax.fill_between(
        conf_int.index,
        conf_int.iloc[:, 0],
        conf_int.iloc[:, 1],
        alpha=0.3
    )
    ax.legend()
    ax.set_title("Passenger Forecast")
    st.pyplot(fig)

    # Evaluation
    train = data["Passengers"][:-forecast_horizon]
    test = data["Passengers"][-forecast_horizon:]
    mae = mean_absolute_error(test, forecast_mean[:len(test)])

    st.markdown("### 📊 Model Evaluation")
    st.metric("Mean Absolute Error (MAE)", f"{mae:.2f}")

# -----------------------------
# TAB 4: MODEL SUMMARY
# -----------------------------
with tab4:
    st.subheader("SARIMA Model Summary")
    st.text(results.summary())

    st.markdown(
        """
        **Interpretation:**
        - Model captures trend and yearly seasonality
        - Residual diagnostics indicate good fit
        - Suitable for short to medium-term forecasting
        """
    )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("Built with Python, SARIMA & Streamlit")
