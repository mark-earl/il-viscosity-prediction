import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

# Load the dataset
dataset_path = 'data/working-dataset-v3.xlsx'
data = pd.read_excel(dataset_path)

# Remove rows where "a" or "b" are missing
data = data.dropna(subset=['a', 'b'])

# Select functional group columns (after "Viscosity Group" column)
functional_groups = data.columns[data.columns.get_loc("Viscosity Group") + 1:]  # assumes functional groups start after 'b'

# Function to run the model for different configurations
def run_random_forest(data, feature_cols, target_col):
    # Prepare features (X) and target (y)
    X = data[feature_cols].values  # Use the values under the functional group columns
    y = data[target_col].values   # Use the target column values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Initialize and train the Random Forest Regressor
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate model
    r2 = r2_score(y_test, y_pred)

    return r2

# Run the experiment 10 times and record the results
n_runs = 10
r2_scores_a = []
r2_scores_b = []

for i in range(n_runs):
    # Run 1: Predict 'b' with functional groups as features
    feature_cols_1 = list(functional_groups)  # Only functional groups as features
    target_col_1 = 'b'
    r2_b = run_random_forest(data, feature_cols_1, target_col_1)
    r2_scores_b.append(r2_b)

    # Run 2: Predict 'a' with functional groups as features
    feature_cols_2 = list(functional_groups)  # Only functional groups as features
    target_col_2 = 'a'
    r2_a = run_random_forest(data, feature_cols_2, target_col_2)
    r2_scores_a.append(r2_a)

# Calculate and print average R^2 scores
avg_r2_b = np.mean(r2_scores_b)
avg_r2_a = np.mean(r2_scores_a)

print(f"Average R^2 over {n_runs} runs:")
print(f"Predicting 'b' with functional groups as features: {avg_r2_b}")
print(f"Predicting 'a' with functional groups as features: {avg_r2_a}")
