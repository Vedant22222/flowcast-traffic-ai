import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configure the page layout
st.set_page_config(page_title="FlowCast Dashboard", layout="wide")

@st.cache_resource
def load_assets():
    # Absolute paths matching your local environment
    data_path = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\processed\analysis_ready_data.csv"
    model_path = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\models\xgboost.pkl"
    scaler_path = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\models\scaler.pkl"
    
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return df, model, scaler

# Load the models and data into memory
df, model, scaler = load_assets()

# Dashboard Header
st.title("🚦 FlowCast: Smart Traffic Intelligence")

# Navigation Tabs mapping to PRD requirements
tab1, tab2, tab3 = st.tabs(["Live Predictions", "Historical Analytics", "Model Scoreboard"])

with tab1:
    st.header("Next-Window Volume Forecast")
    
    # User selects a road segment
    road_selected = st.selectbox("Select Road Segment", df['road_id'].unique())
    
    # Fetch the most recent data row for the selected road to simulate a "live" feed
    road_data = df[df['road_id'] == road_selected].iloc[-1:]
    
    # The exact features the model was trained on
    features = [
        'hour_sin', 'hour_cos', 'is_weekend', 'is_peak_hour',
        'is_raining', 'low_visibility', 
        'vol_lag_1', 'vol_lag_2', 'vol_lag_48',
        'speed_lag_1', 'speed_lag_2', 'vol_roll_mean_4'
    ]
    
    # Standardize the input and predict
    X_input = scaler.transform(road_data[features])
    prediction = model.predict(X_input)[0]
    
    # Display the metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Predicted Volume (Next 30m)", f"{int(prediction)} vehicles")
    col2.metric("Current Weather", road_data['weather_condition'].values[0])
    col3.metric("Last Recorded Volume", f"{int(road_data['traffic_volume'].values[0])} vehicles")

with tab2:
    st.header("Corridor Traffic Trends")
    st.write("Average hourly traffic volume across all segments.")
    # Calculate and plot average traffic per hour
    hourly_trends = df.groupby('hour')['traffic_volume'].mean()
    st.line_chart(hourly_trends)

with tab3:
    st.header("Model Performance Scoreboard")
    st.write("Test set evaluation metrics across algorithms.")
    st.markdown("""
    | Model | RMSE | MAPE (%) |
    |---|---|---|
    | **XGBoost (Active)** | 45.44 | 7.55 |
    | Random Forest | 49.03 | 7.83 |
    | Custom NumPy LR | 72.60 | 13.13 |
    """)