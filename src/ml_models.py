import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error

# ==========================================
# PRD MANDATE: From-Scratch NumPy Regression
# ==========================================
class CustomLinearRegression:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.lr = learning_rate
        self.iterations = iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent Math
        for _ in range(self.iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            # Derivatives
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# ==========================================
# Pipeline Execution
# ==========================================
def train_and_evaluate():
    print("1. Loading processed data...")
    data_path = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\data\processed\analysis_ready_data.csv"
    df = pd.read_csv(data_path)

    # Sort strictly by time before splitting
    df = df.sort_values(by=['datetime'])

    print("2. Defining features and target...")
    # Select our engineered features
    features = [
        'hour_sin', 'hour_cos', 'is_weekend', 'is_peak_hour',
        'is_raining', 'low_visibility', 
        'vol_lag_1', 'vol_lag_2', 'vol_lag_48',
        'speed_lag_1', 'speed_lag_2', 'vol_roll_mean_4'
    ]
    target = 'traffic_volume'

    X = df[features].values
    y = df[target].values

    print("3. Performing time-based split (80% Train, 20% Test)...")
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print("4. Scaling features to prevent magnitude dominance...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\n5. Training Models...")
    models = {
        "Custom NumPy LR": CustomLinearRegression(learning_rate=0.1, iterations=500),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    }

    results = []
    
    # Create models directory
    model_dir = r"C:\Users\Vedant\OneDrive\Desktop\flowcast\models"
    os.makedirs(model_dir, exist_ok=True)

    for name, model in models.items():
        print(f"   -> Training {name}...")
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        
        # Calculate evaluation metrics
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds) * 100
        
        results.append({"Model": name, "RMSE": round(rmse, 2), "MAPE (%)": round(mape, 2)})
        
        # Save library models
        if name != "Custom NumPy LR":
            joblib.dump(model, os.path.join(model_dir, f"{name.replace(' ', '_').lower()}.pkl"))

    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))

    print("\n==============================")
    print("      MODEL SCOREBOARD        ")
    print("==============================")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("==============================\n")
    print(f"Models and scaler saved to {model_dir}")

if __name__ == "__main__":
    train_and_evaluate()