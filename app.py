import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

# -----------------------------
# 🎨 STYLING & CONFIG
# -----------------------------
st.set_page_config(page_title="SkyCast | Passenger Forecasting", page_icon="✈️", layout="wide")

def local_css():
    st.markdown("""
        <style>
        .main { background-color: #f5f7f9; }
        .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        div[data-testid="stExpander"] { border: none !important; box-shadow: 0 2px 4px rgba(0,0,0,0.05); background: white; }
        .stTabs [data-baseweb="tab-list"] { gap: 8px; }
        .stTabs [data-baseweb="tab"] { 
            background-color: #ffffff; border-radius: 4px 4px 0px 0px; padding: 10px 20px;
        }
        </style>
    """, unsafe_allow_html=True)

local_css()

# -----------------------------
# 📊 DATA LOGIC
# -----------------------------
@st.cache_data
def load_and_prep_data():
    df = pd.read_csv("AirPassengers.csv")
    df["Month"] = pd.to_datetime(df["Month"])
    df.set_index("Month", inplace=True)
    df.columns = ["Passengers"]
    return df

def run_forecast(data, horizon):
    # Log transform for variance stabilization
    log_data = np.log(data["Passengers"])
    
    model = SARIMAX(
        log_data,
        order=(1, 1, 1),
        seasonal_order=(1, 1, 1, 12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    results = model.fit(disp=False)
    forecast_obj = results.get_forecast(steps=horizon)
    
    # Back-transforming results
    mean = np.exp(forecast_obj.predicted_mean)
    conf_int = np.exp(forecast_obj.conf_int())
    
    return results, mean, conf_int

# -----------------------------
# ✈️ MAIN APP
# -----------------------------
def main():
    # Header Section
    col_t1, col_t2 = st.columns([1, 4])
    with col_t1:
        st.image("https://cdn-icons-png.flaticon.com/512/784/784844.png", width=80)
    with col_t2:
        st.title("SkyCast: Airline Demand Intelligence")
        st.markdown("*Advanced Seasonal ARIMA Forecasting Engine*")

    # Sidebar
    st.sidebar.header("🕹️ Control Panel")
    forecast_horizon = st.sidebar.slider("Forecast Horizon (Months)", 6, 48, 24)
    
    data = load_and_prep_data()

    # Model Training
    with st.spinner("Refining SARIMA parameters..."):
        results, forecast_mean, conf_int = run_forecast(data, forecast_horizon)

    # 1. KPI Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    last_val = data["Passengers"].iloc[-1]
    forecast_final = forecast_mean.iloc[-1]
    
    m1.metric("Current Volume", f"{int(last_val)}", "Last Obs")
    m2.metric("Projected (End of Horizon)", f"{int(forecast_final)}", f"{((forecast_final/last_val)-1)*100:.1f}%")
    
    # Simple MAE for the last year
    test_actuals = data["Passengers"][-12:]
    test_preds = forecast_mean[:len(test_actuals)]
    mae = mean_absolute_error(test_actuals, test_preds)
    rmse = np.sqrt(mean_squared_error(test_actuals, test_preds))
    
    m3.metric("Model MAE", f"{mae:.2f}")
    m4.metric("Model RMSE", f"{rmse:.2f}")

    st.markdown("---")

    # 2. Main Visuals Tab
    tab_viz, tab_stat, tab_raw = st.tabs(["📈 Forecast Intelligence", "🧪 Statistical Validity", "📂 Data Explorer"])

    with tab_viz:
        fig = go.Figure()
        
        # Historical Data
        fig.add_trace(go.Scatter(x=data.index, y=data["Passengers"], name="Historical", line=dict(color="#1f77b4", width=2)))
        
        # Forecast Mean
        fig.add_trace(go.Scatter(x=forecast_mean.index, y=forecast_mean, name="Forecast", line=dict(color="#FF4B4B", width=3, dash='dot')))
        
        # Confidence Interval
        fig.add_trace(go.Scatter(
            x=list(conf_int.index) + list(conf_int.index)[::-1],
            y=list(conf_int.iloc[:, 1]) + list(conf_int.iloc[:, 0])[::-1],
            fill='toself',
            fillcolor='rgba(255, 75, 75, 0.1)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo="skip",
            name="95% Confidence"
        ))

        fig.update_layout(
            title="Projected Passenger Demand",
            xaxis_title="Date",
            yaxis_title="Total Passengers",
            hovermode="x unified",
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab_stat:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("### Stationarity Test")
            adf = adfuller(data["Passengers"])
            st.info(f"**ADF Statistic:** `{adf[0]:.3f}`\n\n**P-Value:** `{adf[1]:.3f}`")
            if adf[1] > 0.05:
                st.warning("⚠️ Data is Non-Stationary. SARIMA is applying integrated (d=1) differencing.")
            
with col_s2:
            st.markdown("### Residual Analysis")
            # Create the DataFrame
            p_values = results.pvalues.to_frame(name="P-Values")
            
            # Simple highlighting: Significant values (p < 0.05) will be bold/colored
            def color_significant(val):
                color = '#d4edda' if val < 0.05 else 'white'
                return f'background-color: {color}'

            # Use .style.applymap for per-cell styling
            styled_df = p_values.style.applymap(color_significant)
            
            st.dataframe(styled_df, use_container_width=True)

    # Footer
    st.markdown("""<div style='text-align: center; color: grey; padding: 20px;'>SkyCast Analytics v1.0 | © 2026 Airline Data Science Team</div>""", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

