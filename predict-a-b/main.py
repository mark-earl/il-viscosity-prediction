import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Load the dataset
dataset_path = 'data/updated-working-dataset.xlsx'
data = pd.read_excel(dataset_path)

# Remove rows where "a" or "b" are missing
data = data.dropna(subset=['a', 'b'])

# Select functional group columns (after "b" column)
functional_groups = data.columns[data.columns.get_loc("b") + 1:]  # assumes functional groups start after 'b'

# Function to run the model for different configurations
def run_random_forest(data, feature_cols, target_col):
    # Prepare features (X) and target (y)
    X = data[feature_cols]
    y = data[target_col]

    # Convert all column names to strings to avoid sklearn TypeError
    X.columns = X.columns.astype(str)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return mse, r2, mae

# Run 1: Predict 'b' with functional groups and 'a' as features
feature_cols_1 = list(functional_groups) + ['a']
target_col_1 = 'b'
mse_1, r2_1, mae_1 = run_random_forest(data, feature_cols_1, target_col_1)
print(f"Run 1 - Predicting 'b' with 'a' and functional groups as features:")
print(f"MSE: {mse_1}, R^2: {r2_1}, MAE: {mae_1}")

# Run 2: Predict 'a' with functional groups and 'b' as features
feature_cols_2 = list(functional_groups) + ['b']
target_col_2 = 'a'
mse_2, r2_2, mae_2 = run_random_forest(data, feature_cols_2, target_col_2)
print(f"Run 2 - Predicting 'a' with 'b' and functional groups as features:")
print(f"MSE: {mse_2}, R^2: {r2_2}, MAE: {mae_2}")
