F1 Race Winner Prediction APIA machine learning API built with Python and FastAPI that predicts Formula 1 race winners based on historical data and driver form.This backend service powers the F1 Predictor application. It includes scripts for data collection, feature engineering, model training, and a web server to serve the predictions.Core FeaturesData Pipeline: Fetches historical F1 race data from 2018 to the present using the FastF1 library.Feature Engineering: Creates insightful features like driver/constructor form (rolling averages) and qualifying vs. race performance.ML Model: Trains a highly effective XGBoost classifier to predict the win probability for each driver in a race.REST API: Exposes clean, interactive endpoints to fetch race schedules and deliver winner predictions.Tech StackWeb Framework: FastAPIMachine Learning: XGBoost, Scikit-learnData Manipulation: Pandas, FastF1API Server: UvicornProject Structuref1_prediction_project/
├── f1_env/                   # Python Virtual Environment (ignored by git)
├── fastf1_cache/             # Cache for FastF1 data (ignored by git)
├── src/                      # Source scripts
│   ├── data_fetcher.py       # Script to download raw F1 data
│   ├── feature_engineering.py# Script to create ML features
│   └── model_trainer.py      # Script to train and save the XGBoost model
├── .gitignore                # Specifies files for Git to ignore
├── f1_winner_model.json      # The trained and saved ML model
├── main.py                   # The main FastAPI application file
└── requirements.txt          # Project dependencies
Setup and InstallationPrerequisitesGitPython 3.11+1. Clone the Repositorygit clone [https://github.com/your-username/f1-prediction-api.git](https://github.com/your-username/f1-prediction-api.git)
cd f1-prediction-api
2. Create and Activate a Virtual Environment# Create the virtual environment
python3 -m venv f1_env

# Activate it (on Linux/macOS)
source f1_env/bin/activate
3. Install DependenciesFirst, create a requirements.txt file from your local environment (if you haven't already):pip freeze > requirements.txt
Then, install the required packages:pip install -r requirements.txt
UsageFollow these steps in order to set up the model and run the server.1. Fetch Data and Prepare for MLRun the scripts to download data, create features, and train the model.# Step 1: Download raw data (creates f1_fastf1_data.csv)
python3 src/data_fetcher.py

# Step 2: Engineer features (creates f1_ml_ready_data.csv)
python3 src/feature_engineering.py

# Step 3: Train the model (creates f1_winner_model.json)
python3 src/model_trainer.py
2. Run the API ServerOnce the model is trained and saved, start the backend server.uvicorn main:app --reload
The API will now be running on http://127.0.0.1:8000.API EndpointsThe API provides interactive documentation. Once the server is running, navigate to http://127.0.0.1:8000/docs to test the endpoints.Get Race ScheduleEndpoint: GET /schedule/{year}Description: Returns the official F1 race schedule for a given year.Example Request: http://127.0.0.1:8000/schedule/2024Example Response:{
  "schedule": [
    { "EventName": "Bahrain Grand Prix", "Location": "Sakhir" },
    { "EventName": "Saudi Arabian Grand Prix", "Location": "Jeddah" },
    // ...
  ]
}
Get Race Winner PredictionEndpoint: GET /predict/{year}/{race_name}Description: Predicts the top 3 contenders with the highest win probability for a specific race.Example Request: http://127.0.0.1:8000/predict/2024/Italian Grand PrixExample Response:{
  "prediction": [
    { "driver": "VER", "win_probability": 0.6512 },
    { "driver": "LEC", "win_probability": 0.1987 },
    { "driver": "NOR", "win_probability": 0.0855 }
  ]
}
