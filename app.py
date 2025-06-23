
import streamlit as st
import pickle
import numpy as np
import pandas as pd

from preprocessing import preprocess_consumption_data

# Load model
with open('xgb_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load mapping CSV to use in Streamlit
mapping_df = pd.read_csv('local_authority_mapping.csv')

st.title("Electricity Consumption Prediction App")
st.write("Predict total electricity consumption (GWh) for a UK region based on input features.")

# Select Local Authority
local_authority_list = sorted(mapping_df['Local authority'].unique())
local_authority = st.selectbox("Local Authority", local_authority_list)

# Automatically get region and weather zone
region = mapping_df[mapping_df['Local authority'] == local_authority]['Region'].values[0]
weather_zone = mapping_df[mapping_df['Local authority'] == local_authority]['Weather_zone'].values[0]

# Display as greyed-out (disabled) inputs
st.text_input("Region", value=region, disabled=True)
st.text_input("Weather Zone", value=weather_zone, disabled=True)

# Input numeric fields
population = st.number_input("Population", min_value=10000, max_value=10000000, value=500000)
gdp = st.number_input("GDP (£m)", min_value=100, max_value=500000, value=50000)
avg_rainfall = st.number_input("Average Annual Rainfall (mm)", min_value=0.0, max_value=3000.0, value=800.0)
avg_mean_temp = st.number_input("Average Mean Temperature (°C)", min_value=-5.0, max_value=50.0, value=10.0)
total_meters = st.number_input("Total Number of Installed Meters (K)", min_value=10.0, max_value=500.0, value=100.0)
year = st.number_input("Year", min_value=2000, max_value=2030, value=2023)

# Compute derived features
# gdp_actual = gdp * 1e6  # Convert £m to £
gdp_per_capita = round(gdp / population, 6)

# Create DataFrame for model
input_df = pd.DataFrame({
    'Region': [region],
    'Local authority': [local_authority],
    'Total_meters(K)': [total_meters],
    'Year': [year],
    'Weather_zone': [weather_zone],
    'Avg_rainfall': [avg_rainfall],
    'Avg_mean_temp': [avg_mean_temp],
    'GDP': [gdp],  # pass actual GDP in pounds
    'Population': [population],
    'GDP_per_capita': [gdp_per_capita],
})

# Preprocess and predict
try:
    processed_df, _ = preprocess_consumption_data(input_df, training=False)
except FileNotFoundError:
    st.error("Encoders not found. Please train the model first.")
    st.stop()

input_data = processed_df.values

if st.button("Predict"):
    prediction = model.predict(input_data)[0]

    # Display main result
    st.success(
        f"The estimated total electricity consumption for **{local_authority}** in **{year}** is **{prediction:.2f} GWh**."
    )

    # Display uncertainty note
    st.info("Note: This estimate has an expected margin of ±10%.")
    
    # Show processed data for transparency
    st.subheader("Processed Input Data")
    st.dataframe(processed_df)


