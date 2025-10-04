from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import xgboost as xgb
import fastf1 as ff1

# --- App Initialization ---
app = FastAPI()

# --- CORS Middleware ---
# This allows our React frontend (running on a different port) to communicate with this backend.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# --- Model and Data Loading ---
# Load the trained model once when the server starts.
MODEL_PATH = 'f1_winner_model.json'
model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)

# Load the feature-engineered data to get the most recent 'form' for each driver.
DATA_PATH = 'f1_ml_ready_data.csv'
historical_data = pd.read_csv(DATA_PATH)

# Enable FastF1 cache
ff1.Cache.enable_cache('fastf1_cache')


# --- API Endpoint ---
@app.get("/predict/{year}/{race_name}")
async def predict_winner(year: int, race_name: str):
    """
    Predicts the top 3 contenders for a given race.
    
    Args:
        year (int): The year of the race.
        race_name (str): The name of the race (e.g., 'Italian Grand Prix').
    """
    print(f"Received prediction request for {race_name} {year}")
    
    try:
        # --- Step 1: Get session and qualifying data for the requested race ---
        session = ff1.get_session(year, race_name, 'Q') # 'Q' for Qualifying
        session.load(laps=False, telemetry=False, weather=False, messages=False)
        
        # Get the qualifying results to find the starting grid
        qualifying_results = session.results
        
        # --- Step 2: Prepare the feature set for every driver on the grid ---
        prediction_features = []
        driver_ids = qualifying_results['Abbreviation'].unique()

        for driver_id in driver_ids:
            # Find the most recent data for this driver from our historical dataset
            driver_historical = historical_data[historical_data['driverId'] == driver_id].sort_values(by='season', ascending=False).iloc[0]
            
            # Get the grid position from the new qualifying data
            grid_pos = qualifying_results[qualifying_results['Abbreviation'] == driver_id]['Position'].iloc[0]

            # Assemble the features for the model
            features = {
                'grid': grid_pos,
                'driver_form_points': driver_historical['driver_form_points'],
                'driver_form_position': driver_historical['driver_form_position'],
                'constructor_form_points': driver_historical['constructor_form_points'],
                'avg_positions_gained': driver_historical['avg_positions_gained']
            }
            
            prediction_features.append(features)

        # --- Step 3: Make Predictions ---
        df_predict = pd.DataFrame(prediction_features)
        
        # Use predict_proba to get the probability of winning (class 1)
        win_probabilities = model.predict_proba(df_predict)[:, 1]
        
        # --- Step 4: Format the Output ---
        results = []
        for i, driver_id in enumerate(driver_ids):
            results.append({
                'driver': driver_id,
                'win_probability': round(float(win_probabilities[i]), 4)
            })
            
        # Sort by probability to find the top contenders
        results.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return {"prediction": results[:3]} # Return the top 3

    except Exception as e:
        return {"error": f"Could not process the request: {e}"}

@app.get("/schedule/{year}")
async def get_schedule(year: int):
    """
    Returns the race schedule for a given year.
    """
    try:
        schedule = ff1.get_event_schedule(year, include_testing=False)
        races = schedule[['EventName', 'Location']].to_dict(orient='records')
        return {"schedule": races}
    except Exception as e:
        return {"error": f"Could not fetch schedule: {e}"}