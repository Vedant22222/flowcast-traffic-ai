import pandas as pd
import numpy as np
import os

def engineer_features(df):
    print("1. Creating combined datetime and sorting...")
    # Essential to prevent data leakage between roads or from the future
    df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str))
    df = df.sort_values(by=['road_id', 'datetime']).reset_index(drop=True)

    print("2. Engineering temporal features...")
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Cyclical hour encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Peak hour flag (Assuming 7-9 AM and 4-7 PM)
    df['is_peak_hour'] = df['hour'].isin([7, 8, 9, 16, 17, 18, 19]).astype(int)

    print("3. Engineering capacity and weather flags...")
    df['v_c_ratio'] = df['traffic_volume'] / (df['road_capacity'] / 2)
    df['capacity_headroom'] = (df['road_capacity'] / 2) - df['traffic_volume']
    
    df['is_raining'] = (df['weather_condition'] == 'Rain').astype(int)
    df['low_visibility'] = (df['visibility'] < 1000).astype(int) # Assuming < 1km is low

    print("4. Engineering lag and rolling features...")
    # Lag 1 (30 mins ago), Lag 2 (60 mins ago), Lag 48 (Exact same time yesterday)
    for lag in [1, 2, 48]:
        df[f'vol_lag_{lag}'] = df.groupby('road_id')['traffic_volume'].shift(lag)
        df[f'speed_lag_{lag}'] = df.groupby('road_id')['avg_speed'].shift(lag)
        
    # Rolling means and standard deviations (previous 4 windows = 2 hours)
    df['vol_roll_mean_4'] = df.groupby('road_id')['traffic_volume'].transform(lambda x: x.rolling(4, min_periods=1).mean())
    df['speed_roll_mean_4'] = df.groupby('road_id')['avg_speed'].transform(lambda x: x.rolling(4, min_periods=1).mean())
    
    print("5. Cleaning up shifting artifacts...")
    # Shifting creates NaNs at the start of every road's timeline. 
    # We must drop these early rows because ML algorithms cannot train on NaN features.
    df = df.dropna(subset=['vol_lag_48'])

    print(f"Feature engineering complete. Final shape: {df.shape}")
    return df

if __name__ == "__main__":
    # Load the cleaned data from M2
    interim_path = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\interim\cleaned_merged_data.csv"
    print(f"Loading cleaned data from {interim_path}...")
    
    df_cleaned = pd.read_csv(interim_path)
    
    # Run the feature pipeline
    df_features = engineer_features(df_cleaned)
    
    # Save to the processed directory as mandated by PRD
    output_dir = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\processed"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, "analysis_ready_data.csv")
    df_features.to_csv(output_path, index=False)
    print(f"\nSuccess! Analysis-ready data saved to {output_path}")