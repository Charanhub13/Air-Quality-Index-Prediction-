import pandas as pd
import joblib
import streamlit as st

# Load the trained model and scaler
rf_model = joblib.load('rf_model.pkl')
scaler = joblib.load('scaler.pkl')


# Load the data to get feature names
data = pd.read_csv('air_quality_cleaned.csv')


# Streamlit app title with styled header
st.markdown("<h1 style='text-align: center; color: teal;'>🌱 Air Quality Index (AQI) Prediction App 🌱</h1>", unsafe_allow_html=True)

# Sidebar styling
st.markdown("<h2 style='color: navy;'>🌤️ Input Air Quality Parameters</h2>", unsafe_allow_html=True)

# Function to get user input split into two parts
def get_user_input():
    feature_names = data.drop(columns=['City', 'Date', 'AQI', 'AQI_Bucket'], errors='ignore').columns
    user_data = {}
    col1, col2 = st.columns(2)
    for i, feature in enumerate(feature_names):
        if i % 2 == 0:
            user_data[feature] = col1.number_input(feature, min_value=0.0, max_value=100.0, value=0.0)
        else:
            user_data[feature] = col2.number_input(feature, min_value=0.0, max_value=100.0, value=0.0)
    return pd.DataFrame(user_data, index=[0])

# Get user input
user_input = get_user_input()

# Standardize the user input (scaling)
scaled_input = scaler.transform(user_input)

# Make prediction with the Random Forest model
prediction = rf_model.predict(scaled_input)

# Display the prediction with enhanced visuals
st.markdown("<h3 style='text-align: center; color: green;'>🌿 Predicted AQI Category 🌿</h3>", unsafe_allow_html=True)

# Adding color to prediction output
color_map = {
    'Good': 'green',
    'Moderate': 'orange',
    'Poor': 'red',
    'Very Poor': 'purple',
    'Severe': 'brown'
}

predicted_color = color_map.get(prediction[0], 'black')
st.markdown(f"<h2 style='text-align: center; color: {predicted_color};'>{prediction[0]}</h2>", unsafe_allow_html=True)     give source code for this delete streamlit 