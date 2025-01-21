import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Paths to the datasets
working_dataset_path = 'data/working-dataset-v3.xlsx'
raw_dataset_path = 'data/raw.xlsx'

# Load the working dataset
data = pd.read_excel(working_dataset_path)

# Remove rows where "a" or "b" are missing
data = data.dropna(subset=['a', 'b'])

# Load the sheet "S9 | Reference model - SWMLR" from the raw dataset
reference_model = pd.read_excel(raw_dataset_path, sheet_name="S9 | Reference model - SWMLR", header=1)

# Filter rows where "In model" is TRUE
included_groups = reference_model[reference_model["In model"] == True]["Group"]

# Ensure the functional groups are valid column names
included_groups = included_groups.dropna().astype(str).tolist()

# Subset the working dataset to include only these functional groups
functional_groups = [col for col in data.columns if col in included_groups]
data_subset = data[["a", "b"] + functional_groups]

# Normalize the functional group features
scaler = StandardScaler()
data_subset[functional_groups] = scaler.fit_transform(data_subset[functional_groups])

# Define a function to run Random Forest with hyperparameter tuning
def run_random_forest_with_tuning(data, feature_cols, target_col, n_folds=5):
    # Prepare features (X) and target (y)
    X = data[feature_cols].values
    y = data[target_col].values

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define parameter grid for tuning
    param_grid = {
        'n_estimators': [100, 500, 1000],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 5],
        'max_features': ['auto', 'sqrt', 'log2']
    }

    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(random_state=42)

    # Perform grid search cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=n_folds, scoring='r2', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Get the best model and its performance
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    return best_model, r2, grid_search.best_params_

# Run the experiment for 'a' and 'b'
n_runs = 10
r2_scores_a = []
r2_scores_b = []
best_params_a = []
best_params_b = []

for i in range(n_runs):
    print(f"Run {i + 1}/{n_runs}:")

    # Run for predicting 'b'
    print("Predicting 'b'...")
    feature_cols_b = list(functional_groups)
    target_col_b = 'b'
    model_b, r2_b, params_b = run_random_forest_with_tuning(data_subset, feature_cols_b, target_col_b)
    r2_scores_b.append(r2_b)
    best_params_b.append(params_b)
    print(f"R² (b): {r2_b}")

    # Run for predicting 'a'
    print("Predicting 'a'...")
    feature_cols_a = list(functional_groups)
    target_col_a = 'a'
    model_a, r2_a, params_a = run_random_forest_with_tuning(data_subset, feature_cols_a, target_col_a)
    r2_scores_a.append(r2_a)
    best_params_a.append(params_a)
    print(f"R² (a): {r2_a}")

# Calculate average R² scores
avg_r2_b = np.mean(r2_scores_b)
avg_r2_a = np.mean(r2_scores_a)

print("\nAverage R² over multiple runs:")
print(f"Predicting 'b' with functional groups as features: {avg_r2_b}")
print(f"Predicting 'a' with functional groups as features: {avg_r2_a}")

print("\nBest parameters for each run (b):", best_params_b)
print("Best parameters for each run (a):", best_params_a)

# Average R² over multiple runs:
# Predicting 'b' with functional groups as features: -5.929924533183902
# Predicting 'a' with functional groups as features: -6.054161821300946

# Best parameters for each run (b): [{'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}]
# Best parameters for each run (a): [{'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}, {'max_depth': 10, 'max_features': 'log2', 'min_samples_leaf': 5, 'min_samples_split': 2, 'n_estimators': 1000}]
