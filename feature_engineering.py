import pandas as pd

def create_features(input_path='f1_fastf1_data.csv', output_path='f1_ml_ready_data.csv'):
    print(f"Loading raw data from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: '{input_path}' not found.")
        print("Please run data_fetcher.py first to download the raw F1 data.")
        return

    # Sort chronologically to ensure rolling windows work correctly
    df = df.sort_values(by=['season', 'round', 'grid'])

    # --- Target Variable ---
    # 1 if the driver won the race, 0 otherwise
    df['is_winner'] = (df['position'] == 1).astype(int)

    # Positions gained in this race (positive = moved up, negative = dropped back)
    df['positions_gained'] = df['grid'] - df['position']

    # --- Driver Features ---
    print("Engineering driver form features (rolling 3-race averages)...")
    driver_group = df.groupby('driverId')
    
    # We use shift(1) so the current race's results aren't included in the prediction features
    df['driver_form_points'] = driver_group['points'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
    )
    df['driver_form_position'] = driver_group['position'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['avg_positions_gained'] = driver_group['positions_gained'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )

    # --- Constructor Features ---
    print("Engineering constructor form features...")
    # First, aggregate points by constructor per race (since teams have 2 drivers)
    constructor_group = df.groupby(['season', 'round', 'constructorId'])['points'].sum().reset_index()
    constructor_group = constructor_group.sort_values(by=['season', 'round'])
    
    # Calculate the rolling sum for constructors
    constructor_group['constructor_form_points'] = constructor_group.groupby('constructorId')['points'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).sum()
    )
    
    # Merge the constructor form back into the main dataframe
    df = pd.merge(
        df, constructor_group[['season', 'round', 'constructorId', 'constructor_form_points']], 
        on=['season', 'round', 'constructorId'], how='left'
    )

    # --- Cleanup ---
    # Fill missing values (which happen on a driver/constructor's very first race) with 0
    df.fillna(0, inplace=True)

    print(f"Saving ML-ready data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("✅ Feature engineering complete!")

if __name__ == "__main__":
    create_features()