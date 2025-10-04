import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

def train_model(data_path='f1_ml_ready_data.csv'):
    """
    Trains an XGBoost model to predict F1 race winners and saves it.

    Args:
        data_path (str): Path to the machine learning-ready data.
    """
    print("Loading ML-ready data...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: The file {data_path} was not found.")
        print("Please run the feature_engineering.py script first.")
        return

    # --- Feature Selection ---
    # Define which columns are our features (X) and which is our target (y)
    # We exclude identifiers that are not predictive features.
    features = [
        'grid', 'driver_form_points', 'driver_form_position',
        'constructor_form_points', 'avg_positions_gained'
    ]
    target = 'is_winner'

    X = df[features]
    y = df[target]

    # --- Data Splitting (Chronological) ---
    # For time-series data, we should not split randomly.
    # We'll train on older data and test on the most recent data.
    # Let's use the last 2 seasons for testing.
    max_season = df['season'].max()
    train_df = df[df['season'] < max_season - 1]
    test_df = df[df['season'] >= max_season - 1]

    X_train = train_df[features]
    y_train = train_df[target]
    X_test = test_df[features]
    y_test = test_df[target]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")

    # --- Model Training ---
    print("\nTraining XGBoost model...")
    
    # We use scale_pos_weight to handle the class imbalance,
    # as there are many more losers (0) than winners (1) in each race.
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='logloss',
        use_label_encoder=False,
        n_estimators=100, # Number of trees
        max_depth=3,      # Maximum depth of a tree
        learning_rate=0.1 # Step size shrinkage
    )

    model.fit(X_train, y_train)

    # --- Model Evaluation ---
    print("\nEvaluating model performance...")
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy on Test Set: {accuracy * 100:.2f}%")
    
    # The classification report gives us more details like precision and recall.
    # It will be heavily skewed because of the imbalance, but it's good to see.
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Winner', 'Winner']))

    # --- Save the Model ---
    model_filename = 'f1_winner_model.json'
    model.save_model(model_filename)
    print(f"\n✅ Success! Trained model has been saved to '{model_filename}'.")

# --- Main Execution ---
if __name__ == "__main__":
    train_model()