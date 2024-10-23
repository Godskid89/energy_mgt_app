import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Load your pre-existing data (assuming it's in a CSV or could be from a database)
@st.cache_resource
def load_energy_data():
    # Replace this with your actual data source
    df = pd.read_csv('train_features.csv')  # Assuming you have this file locally
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

def energy_forecast_app():
    st.title("Energy Consumption Forecast for Multiple Buildings")

    # Load the pre-existing energy usage data
    df = load_energy_data()

    # Get the list of unique building IDs from the data
    building_ids = df['building_id'].unique()

    # Allow users to select multiple Building IDs
    selected_building_ids = st.multiselect("Select Building IDs:", building_ids)

    # Check if any building IDs are selected
    if selected_building_ids:
        # Filter data by the selected building IDs
        filtered_data = df[df['building_id'].isin(selected_building_ids)]

        if not filtered_data.empty:
            # Visualize the historical data for the selected buildings
            st.write(f"Historical Energy Usage for Selected Buildings")
            for building_id in selected_building_ids:
                building_data = filtered_data[filtered_data['building_id'] == building_id]
                st.line_chart(building_data.set_index('timestamp')['meter_reading'], height=300)

            # Proceed with energy forecast using Prophet
            forecast_energy_usage(filtered_data, selected_building_ids)
        else:
            st.error("No data found for the selected Building IDs. Please try another selection.")

def forecast_energy_usage(df, building_ids):
    # Create an empty DataFrame to store the combined forecast results
    combined_forecast = pd.DataFrame()

    # Initialize a figure for the combined line chart
    fig, ax = plt.subplots(figsize=(10, 6))

    # Iterate over each building ID and make a forecast
    for building_id in building_ids:
        # Filter data for the specific building
        df_building = df[df['building_id'] == building_id]

        # Prophet expects 'ds' as the date column and 'y' as the value column
        df_prophet = df_building[['timestamp', 'meter_reading']].rename(columns={'timestamp': 'ds', 'meter_reading': 'y'})

        # Initialize and train the Prophet model on the filtered data
        model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
        model.fit(df_prophet)

        # User input for forecast horizon with a unique key for each building
        period = st.slider(f"Select number of months to predict for Building {building_id}", 1, 12, 3, key=f"slider_{building_id}")

        # Create a future dataframe for the forecast
        future = model.make_future_dataframe(periods=period * 30, freq='D')  # Forecast for 'period' months
        forecast = model.predict(future)

        # Add the building ID to the forecast dataframe
        forecast['building_id'] = building_id

        # Append the forecast results to the combined DataFrame
        combined_forecast = pd.concat([combined_forecast, forecast[['ds', 'yhat', 'building_id']]])

        # Plot each building's forecast as a line on the chart
        ax.plot(forecast['ds'], forecast['yhat'], label=f'Building {building_id}')

    # Show the combined forecast results in a table
    st.write("Forecasted Energy Consumption for the Next Period")
    st.dataframe(combined_forecast.tail())

    # Finalize and show the combined line chart
    ax.set_xlabel("Date")
    ax.set_ylabel("Predicted Meter Reading")
    ax.set_title("Energy Consumption Forecast for Selected Buildings")
    ax.legend()
    st.pyplot(fig)

    # Show individual components (trends and seasonality) for the last building selected
    st.write("Forecast Components for the Last Selected Building")
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# Call this function in app.py to render the energy forecast page
