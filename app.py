import streamlit as st
from energy_anomaly_app import energy_anomaly_app
from energy_forecast_app import energy_forecast_app

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Page", ["Anomaly Detection", "Energy Usage Forecast"])

# Render the selected page
if page == "Anomaly Detection":
    energy_anomaly_app()  # Existing anomaly detection app
elif page == "Energy Usage Forecast":
    energy_forecast_app()  # New forecast page
