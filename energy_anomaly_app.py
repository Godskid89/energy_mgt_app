import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier
import joblib

# Load the pre-trained CatBoost model
@st.cache_resource
def load_catboost_model():
    model = joblib.load('catboost_model.pkl')
    return model

def energy_anomaly_app():
    st.title('Household Energy Usage Anomaly Detection')

    # Input method: File upload or manual data input
    input_method = st.radio("How would you like to input data?", ('Upload CSV File', 'Manually Input Data'))

    if input_method == 'Upload CSV File':
        uploaded_file = st.file_uploader("Choose a CSV file with historical data", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            detect_anomalies(df)

    elif input_method == 'Manually Input Data':
        # Define input fields in the correct order
        manual_data = []
        num_entries = st.number_input("How many data entries would you like to input?", min_value=1, max_value=50, value=5)

        for i in range(num_entries):
            timestamp = st.text_input(f"Enter timestamp for entry {i+1} (YYYY-MM-DD HH:MM:SS):", key=f"timestamp_{i}")
            meter_reading = st.number_input(f"Enter meter reading for entry {i+1}:", key=f"meter_reading_{i}")
            air_temperature = st.number_input(f"Enter air temperature for entry {i+1}:", key=f"air_temperature_{i}")
            square_feet = st.number_input(f"Enter square feet for entry {i+1}:", key=f"square_feet_{i}")
            year_built = st.number_input(f"Enter year built for entry {i+1}:", key=f"year_built_{i}")
            floor_count = st.number_input(f"Enter floor count for entry {i+1}:", key=f"floor_count_{i}")
            primary_use = st.text_input(f"Enter primary use (e.g., Education, Office) for entry {i+1}:", key=f"primary_use_{i}")
            sea_level_pressure = st.number_input(f"Enter sea level pressure for entry {i+1}:", key=f"sea_level_pressure_{i}")
            cloud_coverage = st.number_input(f"Enter cloud coverage (0-8) for entry {i+1}:", key=f"cloud_coverage_{i}")
            is_holiday = st.selectbox(f"Is it a holiday for entry {i+1}?", (0, 1), key=f"is_holiday_{i}")
            dew_temperature = st.number_input(f"Enter dew temperature for entry {i+1}:", key=f"dew_temperature_{i}")

            # Time-based features derived from timestamp
            if timestamp:
                dt = pd.to_datetime(timestamp)
                hour = dt.hour
                weekday = dt.weekday()
                day = dt.day
                week = dt.isocalendar().week
                month = dt.month
                year = dt.year
            else:
                hour, weekday, day, week, month, year = None, None, None, None, None, None

            if timestamp and meter_reading:
                manual_data.append({
                    'meter_reading': meter_reading,
                    'air_temperature': air_temperature,
                    'square_feet': square_feet,
                    'year_built': year_built,
                    'floor_count': floor_count,
                    'primary_use': primary_use,
                    'sea_level_pressure': sea_level_pressure,
                    'cloud_coverage': cloud_coverage,
                    'is_holiday': is_holiday,
                    'dew_temperature': dew_temperature,
                    'hour': hour,
                    'weekday': weekday,
                    'day': day,
                    'week': week,
                    'month': month,
                    'year': year
                })

        # If the data is provided, proceed with anomaly detection
        if manual_data:
            df_manual = pd.DataFrame(manual_data)
            detect_anomalies(df_manual)

def detect_anomalies(df):
    # Ensure that the input features are in the same order as the model's training data
    features = ['meter_reading', 'air_temperature', 'square_feet', 'year_built', 'floor_count', 'primary_use',
                'sea_level_pressure', 'cloud_coverage', 'is_holiday', 'dew_temperature', 'hour', 'weekday', 'day',
                'week', 'month', 'year']

    X = df[features]

    # Load the pre-trained model
    model = load_catboost_model()

    # Make predictions using the pre-trained model
    predictions = model.predict(X)

    # Add predictions to the DataFrame (0: Normal, 1: Anomaly)
    df['Anomaly'] = predictions

    # Display the results
    st.write("Anomaly Detection Results:")
    st.write(df[['meter_reading', 'Anomaly']].head())

    # Plot meter reading over time with anomalies highlighted
    st.write("Meter Reading with Anomalies")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['meter_reading'], label='Meter Reading')
    ax.scatter(df[df['Anomaly'] == 1].index, df[df['Anomaly'] == 1]['meter_reading'], color='red', label='Anomalies')
    plt.xlabel('Index')
    plt.ylabel('Meter Reading')
    plt.legend()
    st.pyplot(fig)

    # Display the number of detected anomalies
    anomaly_count = df['Anomaly'].sum()
    st.write(f"Detected {anomaly_count} anomalies.")
