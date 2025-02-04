import os
from data_preprocessing import load_data, preprocess_data, select_features
from model_training import split_data, train_model, evaluate_model
from visualization import plot_results
from feature_importance import calculate_feature_importance, plot_feature_importance
from confidence_interval import calculate_confidence_interval, plot_confidence_interval
from committee_confidence_interval import calculate_committee_confidence_interval, plot_committee_confidence_interval

################################################################################################################################

# SET UP

# Data
DATA_PATH = 'data/xlsx/working-ils.xlsx'

# Features
FEATURE_SET_CHOICE = 'both'  # Options: "functional_groups", "molecular_descriptors", "both"
OVERRIDE_FEATURES = False

# Add a switch to choose between CatBoost-only and committee approach
USE_COMMITTEE = True  # Set to False to use only CatBoost

# Confidence Interval Settings
NUM_RUNS = 50

# Feature Importance
NUM_FEATURES = 35

################################################################################################################################

# Step 1: Load and preprocess data
df = load_data(DATA_PATH)

# Step 2: Preprocess data
included_data, excluded_data = preprocess_data(df)

# Step 3: Select features
X_included = select_features(included_data, FEATURE_SET_CHOICE, OVERRIDE_FEATURES, feature_json_path=r"results\top_35_features\using_top_families\mol_des\feature_importances.json")
y_included = included_data['Reference Viscosity Log']

# Step 4: Train-test split
# Ensure train-test split is stratified by cationic and anionic families
X_train, X_test, y_train, y_test = split_data(X_included, y_included)

# # Step 5: Train model
# model = train_model(X_train, y_train)

# # Step 6: Evaluate model
# y_pred, r2_rand = evaluate_model(model, X_test, y_test)

# # Step 7: Plot results
# plot_results(included_data, excluded_data, y_test, y_pred, r2_rand)

# Optional: Feature importance and confidence interval
# feature_importance_df = calculate_feature_importance(model, X_included, NUM_FEATURES)
# plot_feature_importance(feature_importance_df, NUM_FEATURES)

if USE_COMMITTEE:
    mean_r2, confidence_interval, r2_scores = calculate_committee_confidence_interval(X_included, y_included, NUM_RUNS)
    plot_committee_confidence_interval(r2_scores, confidence_interval, mean_r2)
else:
    mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(X_included, y_included, NUM_RUNS, model_name="mlp")
    plot_confidence_interval(r2_scores, confidence_interval, mean_r2)

# Print results
# print(f"Model R² on test data: {r2_rand:.2f}")
