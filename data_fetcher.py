import pandas as pd
import fastf1 as ff1
import time

def get_f1_data(start_year, end_year):
    """
    Fetches F1 race and qualifying data for a range of years using the FastF1 library.

    Args:
        start_year (int): The first year to fetch data for.
        end_year (int): The last year to fetch data for.

    Returns:
        pandas.DataFrame: A DataFrame containing combined race and qualifying data.
    """
    all_results = []
    
    # Enable the cache to speed up subsequent runs
    ff1.Cache.enable_cache('fastf1_cache') 
    
    print(f"Fetching data from {start_year} to {end_year}...")

    for year in range(start_year, end_year + 1):
        print(f"\nProcessing Season: {year}")
        try:
            # Get the schedule for the year
            schedule = ff1.get_event_schedule(year, include_testing=False)
            
            # Iterate through each race in the schedule
            for i, event in schedule.iterrows():
                # We only want to process actual races, not other session types
                if event['EventFormat'] == 'conventional':
                    try:
                        # Load the race session data
                        session = ff1.get_session(year, event['RoundNumber'], 'R')
                        session.load(laps=False, telemetry=False, weather=False, messages=False)
                        
                        # Get race results
                        results = session.results
                        
                        # Add session-specific info to each driver's result
                        results['season'] = year
                        results['round'] = event['RoundNumber']
                        results['raceName'] = event['EventName']
                        results['circuitId'] = event['Location'] # Using location as a proxy for circuitId
                        results['date'] = event['EventDate'].date()
                        
                        # Rename columns to match our original format
                        results.rename(columns={
                            'DriverNumber': 'driverNumber',
                            'Abbreviation': 'driverId',
                            'TeamName': 'constructorId',
                            'GridPosition': 'grid',
                            'Position': 'position',
                            'Points': 'points',
                            'Status': 'status'
                        }, inplace=True)
                        
                        # Select only the columns we need
                        relevant_cols = [
                            'season', 'round', 'circuitId', 'raceName', 'date',
                            'driverId', 'constructorId', 'grid', 'position',
                            'points', 'status'
                        ]
                        
                        # Append to our main list
                        all_results.append(results[relevant_cols])
                        print(f"  > Fetched data for: {event['EventName']}")
                        
                    except Exception as e:
                        print(f"  > Could not load session for {event['EventName']}: {e}")
                
                # A small delay to be respectful to the API
                time.sleep(1)

        except Exception as e:
            print(f"Could not process year {year}: {e}")

    # Combine all dataframes from the list into a single one
    if not all_results:
        print("No data was fetched. The final DataFrame is empty.")
        return pd.DataFrame()
        
    final_df = pd.concat(all_results, ignore_index=True)
    return final_df

# --- Main Execution ---
if __name__ == "__main__":
    START_YEAR = 2018 # FastF1 is most reliable from 2018 onwards
    END_YEAR = 2025   # Fetches up to the current year
    
    df_data = get_f1_data(START_YEAR, END_YEAR)
    
    if not df_data.empty:
        # Convert data types for consistency
        df_data['grid'] = pd.to_numeric(df_data['grid'], errors='coerce').fillna(0).astype(int)
        df_data['position'] = pd.to_numeric(df_data['position'], errors='coerce').fillna(0).astype(int)
        df_data['points'] = pd.to_numeric(df_data['points'], errors='coerce').fillna(0).astype(float)
        
        output_filename = 'f1_fastf1_data.csv'
        df_data.to_csv(output_filename, index=False)
        
        print(f"\n✅ Success! Data has been saved to '{output_filename}'.")
        print(f"Total rows: {len(df_data)}")
        print("\nPreview of the data:")
        print(df_data.head())