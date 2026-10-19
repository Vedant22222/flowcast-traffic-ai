import pandas as pd
import numpy as np
import json
import os

def clean_and_merge(traffic_path, weather_path, calendar_path):
    print("1. Loading datasets...")
    df_traffic = pd.read_csv(traffic_path)
    df_weather = pd.read_csv(weather_path)
    df_calendar = pd.read_csv(calendar_path)

    print("2. Standardizing dates...")
    df_traffic['date'] = pd.to_datetime(df_traffic['date'], format='%Y-%m-%d')
    df_weather['date'] = pd.to_datetime(df_weather['date'], format='%d/%m/%Y')
    df_calendar['date'] = pd.to_datetime(df_calendar['date'], format='%Y-%m-%d')

    print("3. Removing duplicates in traffic data...")
    df_traffic = df_traffic.drop_duplicates(subset=['road_id', 'date', 'time'], keep='first')

    print("4. Handling physically impossible outliers...")
    df_traffic.loc[df_traffic['traffic_volume'] < 0, 'traffic_volume'] = np.nan
    df_traffic.loc[df_traffic['avg_speed'] > 200, 'avg_speed'] = np.nan
    df_traffic.loc[df_traffic['occupancy'] > 100, 'occupancy'] = np.nan

    print("5. Imputing missing sensor gaps...")
    # Sorting chronologically is required for valid time-series interpolation
    df_traffic = df_traffic.sort_values(['road_id', 'date', 'time'])
    for col in ['traffic_volume', 'avg_speed', 'occupancy']:
        df_traffic[col] = df_traffic.groupby('road_id')[col].transform(
            lambda x: x.interpolate(method='linear').ffill().bfill()
        )

    print("6. Harmonizing weather categories...")
    df_weather['weather_condition'] = df_weather['weather_condition'].str.strip().str.title()
    df_weather['weather_condition'] = df_weather['weather_condition'].replace({'Rainy': 'Rain'})

    print("7. Temporal alignment & merging...")
    df_traffic['hour'] = pd.to_datetime(df_traffic['time'], format='%H:%M').dt.hour
    df_weather['hour'] = pd.to_datetime(df_weather['time'], format='%H:%M').dt.hour

    df_merged = pd.merge(df_traffic, df_weather, how='left', left_on=['weather_station_id', 'date', 'hour'], right_on=['station_id', 'date', 'hour'])
    df_merged = pd.merge(df_merged, df_calendar, how='left', on='date')

    # Cleanup redundant columns
    cols_to_drop = ['station_id', 'hour', 'time_y']
    df_merged = df_merged.drop(columns=[col for col in cols_to_drop if col in df_merged.columns])
    df_merged = df_merged.rename(columns={'time_x': 'time'})

    print("8. Deriving missing congestion levels...")
    v_c_ratio = df_merged['traffic_volume'] / (df_merged['road_capacity'] / 2)
    conditions = [
        v_c_ratio < 0.50,
        (v_c_ratio >= 0.50) & (v_c_ratio < 0.80),
        (v_c_ratio >= 0.80) & (v_c_ratio < 1.00),
        v_c_ratio >= 1.00
    ]
    choices = ['Free-flow', 'Moderate', 'Heavy', 'Severe']
    derived_congestion = pd.Series(np.select(conditions, choices, default=None), index=df_merged.index)
    df_merged['congestion_level'] = df_merged['congestion_level'].fillna(derived_congestion)

    print("9. Parsing JSON vehicle distribution...")
    def parse_dist(x):
        try:
            return pd.Series(json.loads(x.replace("'", '"')))
        except:
            return pd.Series({'2W': 0.0, 'Car': 0.0, 'LCV': 0.0, 'HCV': 0.0})
    
    dist_df = df_merged['vehicle_type_dist'].apply(parse_dist)
    df_merged = pd.concat([df_merged, dist_df], axis=1).drop('vehicle_type_dist', axis=1)

    print(f"Cleaning & Merge complete. Final shape: {df_merged.shape}")
    return df_merged

if __name__ == "__main__":
    # Your verified file locations
    traffic_csv = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\raw\traffic_sensor_log.csv"
    weather_csv = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\raw\weather_observations.csv"
    calendar_csv = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\raw\calendar_events.csv"
    
    cleaned_data = clean_and_merge(traffic_csv, weather_csv, calendar_csv)
    
    # Save the output to the interim folder as per PRD
    output_dir = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\interim"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "cleaned_merged_data.csv")
    cleaned_data.to_csv(output_path, index=False)
    print(f"\nSuccess! Cleaned data saved to {output_path}")