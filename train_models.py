import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import numpy as np
import joblib

# Load the complete, augmented dataset
df = pd.read_csv('augmented_processed_data.csv')

# --- Define features (X) and target (y) ---

# The target variable is 'warehouse_sales'
y = df['warehouse_sales']

# Drop the target variable AND non-feature identifier columns
X = df.drop(columns=[
    'date', 
    'supplier', 
    'item_code', 
    'item_description', 
    'warehouse_sales'
])

print("Features being used for training:", X.columns.tolist())

# Split data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Historical Demand Model: Random Forest ---
print("Training Historical Forecaster (Random Forest)...")
rf = RandomForestRegressor(random_state=42, n_jobs=-1)
rf_model = rf.set_params(n_estimators=100, max_depth=10).fit(X_train, y_train)
joblib.dump(rf_model, 'historical_model.pkl')
print("Historical model training complete.")

# --- Train Counterfactual Model: XGBoost ---
print("Training Counterfactual Forecaster (XGBoost)...")
xgb = XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1)
xgb_model = xgb.set_params(n_estimators=100, max_depth=5, learning_rate=0.1).fit(X_train, y_train)
joblib.dump(xgb_model, 'counterfactual_model.pkl')
print("Counterfactual model training complete.")

# Probabilistic forecast function
def get_probabilistic_forecast(model, data_row):
    if isinstance(model, RandomForestRegressor):
        predictions = [tree.predict(data_row) for tree in model.estimators_]
        return np.mean(predictions), np.std(predictions)
    else:
        prediction = model.predict(data_row)
        return prediction[0], prediction[0] * 0.1 # 10% uncertainty heuristic
        
# Example usage
test_row = X_test.iloc[[0]]
mean_pred, std_dev_pred = get_probabilistic_forecast(rf_model, test_row)
print(f"\nExample Probabilistic Forecast:")
print(f"Predicted Mean Demand: {mean_pred:.2f}, Predicted Std Dev: {std_dev_pred:.2f}")