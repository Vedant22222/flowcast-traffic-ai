import pandas as pd
import numpy as np

def load_and_merge_data(traffic_path, weather_path, calendar_path):
    print("Loading datasets...")
    # 1. Load Raw Data
    df_traffic = pd.read_csv(traffic_path)
    df_weather = pd.read_csv(weather_path)
    df_calendar = pd.read_csv(calendar_path)

    print("Standardizing temporal columns...")
    # 2. Standardize Dates
    # Traffic is YYYY-MM-DD
    df_traffic['date'] = pd.to_datetime(df_traffic['date'], format='%Y-%m-%d')
    # Weather is DD/MM/YYYY
    df_weather['date'] = pd.to_datetime(df_weather['date'], format='%d/%m/%Y')
    # Calendar is YYYY-MM-DD
    df_calendar['date'] = pd.to_datetime(df_calendar['date'], format='%Y-%m-%d')

    # 3. Create Alignment Keys
    # Extract just the hour from traffic time (e.g., '14:30' -> 14) to join with hourly weather
    df_traffic['hour'] = pd.to_datetime(df_traffic['time'], format='%H:%M').dt.hour
    df_weather['hour'] = pd.to_datetime(df_weather['time'], format='%H:%M').dt.hour

    print("Merging tables...")
    # 4. Merge Weather onto Traffic
    # Broadcasts the hourly weather to both 30-min windows in that hour
    df_merged = pd.merge(
        df_traffic,
        df_weather,
        how='left',
        left_on=['weather_station_id', 'date', 'hour'],
        right_on=['station_id', 'date', 'hour']
    )

    # 5. Merge Calendar onto Traffic
    df_merged = pd.merge(
        df_merged,
        df_calendar,
        how='left',
        on='date'
    )

    # 6. Cleanup redundant columns
    columns_to_drop = ['station_id', 'hour', 'time_y'] # time_y comes from weather
    df_merged = df_merged.drop(columns=[col for col in columns_to_drop if col in df_merged.columns])
    df_merged = df_merged.rename(columns={'time_x': 'time'})

    print(f"Merge complete. Final shape: {df_merged.shape}")
    return df_merged

if __name__ == "__main__":
    # Adjust the paths based on your folder structure
    merged_data = load_and_merge_data(
        r"C:\\Users\\Vedant\\OneDrive\\Desktop\\flowcast\\data\\raw\\traffic_sensor_log.csv",
        r"C:\\Users\\Vedant\\OneDrive\\Desktop\\flowcast\\data\\raw\\weather_observations.csv",
        r"C:\\Users\\Vedant\\OneDrive\\Desktop\\flowcast\\data\\raw\\calendar_events.csv"
    )
    
    print("\nSample of merged dataframe:")
    print(merged_data[['road_id', 'date', 'time', 'weather_condition', 'public_holiday']].head())