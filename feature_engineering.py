import pandas as pd

def create_features(data_path='f1_fastf1_data.csv'):
    """
    Loads the raw F1 data and engineers new features for the ML model.

    Args:
        data_path (str): The path to the raw data CSV file.

    Returns:
        pandas.DataFrame: A DataFrame with new, engineered features.
    """
    print("Loading raw data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        print("Please run the data_fetcher.py script first.")
        return None

    # --- Data Cleaning and Sorting ---
    # Sort data chronologically to ensure rolling calculations are correct
    df.sort_values(by=['season', 'round'], inplace=True)
    
    # --- Feature Engineering ---
    print("Engineering new features...")

    # 1. Driver Form (Rolling Averages)
    # Calculate the average points and position over the last 5 races for each driver
    # We use .shift(1) to prevent data leakage from the current race
    df['driver_form_points'] = df.groupby('driverId')['points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(0)
    
    df['driver_form_position'] = df.groupby('driverId')['position'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(0)

    # 2. Constructor Form (Rolling Averages)
    # Calculate average points over the last 5 races for each constructor
    df['constructor_form_points'] = df.groupby('constructorId')['points'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(0)

    # 3. Qualifying vs. Race Performance
    # Calculate positions gained or lost from the start
    # Note: A positive number means the driver moved up the field.
    df['positions_gained'] = df['grid'] - df['position']
    df['avg_positions_gained'] = df.groupby('driverId')['positions_gained'].transform(
        lambda x: x.shift(1).rolling(window=5, min_periods=1).mean()
    ).fillna(0)
    
    # --- Final Touches ---
    # We only care about predicting the winner, so let's create our target variable
    df['is_winner'] = (df['position'] == 1).astype(int)
    
    # Select the columns we will use for the model
    # We include identifiers like season and raceName for context, but won't use them for training
    feature_columns = [
        'season', 'round', 'raceName', 'circuitId', 'driverId', 'constructorId',
        'grid', 'driver_form_points', 'driver_form_position', 
        'constructor_form_points', 'avg_positions_gained', 'is_winner'
    ]
    
    df_final = df[feature_columns].copy()
    
    print("Feature engineering complete.")
    return df_final


# --- Main Execution ---
if __name__ == "__main__":
    ml_ready_data = create_features()
    
    if ml_ready_data is not None:
        output_filename = 'f1_ml_ready_data.csv'
        ml_ready_data.to_csv(output_filename, index=False)
        
        print(f"\n✅ Success! Machine learning-ready data saved to '{output_filename}'.")
        print("\nPreview of the final data:")
        print(ml_ready_data.head())