import fastf1
import fastf1.plotting
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Optional: Enable fastf1 caching
cache_dir = os.path.join(os.path.dirname(__file__), '../data/cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

def fetch_session_data(year, event, session_type):
    """
    Fetches session data from FastF1.
    """
    session = fastf1.get_session(year, event, session_type)
    session.load()
    return session

def process_lap_data(session):
    """
    Extracts relevant lap features and weather data.
    Features: 'LapNumber', 'Stint', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
              'Compound', 'TyreLife', 'TrackTemp', 'AirTemp', 'Humidity'.
    """
    laps = session.laps
    weather = session.weather_data

    # Merge weather data
    # weather data is recorded during the session, we can match it roughly by Time
    # fastf1 laps have a 'Time' column which is session time.
    if weather.empty:
        # Some sessions might miss weather data
        laps['TrackTemp'] = np.nan
        laps['AirTemp'] = np.nan
        laps['Humidity'] = np.nan
    else:
        # Find nearest weather data for each lap
        weather_times = weather['Time']
        laps = laps.assign(
            WeatherIdx=[(weather_times - lap['Time']).abs().argmin() for _, lap in laps.iterrows()]
        )
        laps['TrackTemp'] = weather['TrackTemp'].iloc[laps['WeatherIdx']].values
        laps['AirTemp'] = weather['AirTemp'].iloc[laps['WeatherIdx']].values
        laps['Humidity'] = weather['Humidity'].iloc[laps['WeatherIdx']].values
        laps = laps.drop(columns=['WeatherIdx'])

    return laps

def clean_data(df):
    """
    Cleans the dataframe by handling missing values and converting types.
    """
    # Keep relevant columns
    cols_to_keep = [
        'Driver', 'LapNumber', 'Stint', 'LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'Compound', 'TyreLife', 'TrackTemp', 'AirTemp', 'Humidity'
    ]
    df = df[cols_to_keep].copy()

    # Drop laps without lap time (out laps, in laps sometimes, or invalid laps)
    df = df.dropna(subset=['LapTime'])
    
    # Convert timedelta to seconds
    for col in ['LapTime', 'Sector1Time', 'Sector2Time', 'Sector3Time']:
        df[col] = df[col].dt.total_seconds()
        
    # Fill missing sectors with median or drop
    df = df.dropna(subset=['Sector1Time', 'Sector2Time', 'Sector3Time', 'Compound', 'TyreLife'])
    
    return df

def feature_engineering(df):
    """
    Encodes categorical variables and standardizes numerical ones.
    """
    # Categorical encoding
    le = LabelEncoder()
    df['Compound_Label'] = le.fit_transform(df['Compound'].astype(str))
    
    # Store encoder for decoding if needed later
    # (In a real scenario, you'd save this object)
    
    # Separate numeric columns for scaling
    num_cols = [
        'LapNumber', 'Stint', 'Sector1Time', 'Sector2Time', 'Sector3Time',
        'TyreLife', 'TrackTemp', 'AirTemp', 'Humidity'
    ]
    
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df[num_cols])
    
    for i, col in enumerate(num_cols):
        df[f'{col}_scaled'] = scaled_features[:, i]
        
    return df, scaler, le

def create_sequences(df, sequence_length=5):
    """
    Generates time-series sequences using sliding window.
    Sorts by Driver and LapNumber to maintain sequence integrity.
    """
    df = df.sort_values(['Driver', 'LapNumber'])
    
    feature_cols = [
        'LapNumber_scaled', 'Stint_scaled', 'Sector1Time_scaled', 'Sector2Time_scaled', 
        'Sector3Time_scaled', 'Compound_Label', 'TyreLife_scaled', 
        'TrackTemp_scaled', 'AirTemp_scaled', 'Humidity_scaled'
    ]
    
    target_col = 'LapTime'
    
    X = []
    y = []
    
    for driver in df['Driver'].unique():
        driver_df = df[df['Driver'] == driver].reset_index(drop=True)
        if len(driver_df) <= sequence_length:
            continue
            
        driver_features = driver_df[feature_cols].values
        driver_targets = driver_df[target_col].values
        
        for i in range(len(driver_df) - sequence_length):
            X.append(driver_features[i : i + sequence_length])
            # Target is the lap time of the *next* lap
            y.append(driver_targets[i + sequence_length])
            
    return np.array(X), np.array(y)

def prepare_dataset(year, event, session_type='R', sequence_length=5):
    session = fetch_session_data(year, event, session_type)
    laps_df = process_lap_data(session)
    clean_df = clean_data(laps_df)
    engineered_df, scaler, le = feature_engineering(clean_df)
    X, y = create_sequences(engineered_df, sequence_length)
    return X, y, scaler, le, engineered_df

if __name__ == "__main__":
    # Test execution
    print("Testing data processing module...")
    # X, y, scaler, le, df = prepare_dataset(2023, 'Bahrain', 'R')
    # print(f"X shape: {X.shape}, y shape: {y.shape}")
