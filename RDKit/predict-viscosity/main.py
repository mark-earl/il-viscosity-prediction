import os
from data_preprocessing import load_data, preprocess_data, select_features
from model_training import split_data, train_model, evaluate_model
from visualization import plot_results
from feature_importance import calculate_feature_importance, plot_feature_importance
from confidence_interval import calculate_confidence_interval, plot_confidence_interval

# Paths and configurations
DATA_PATH = 'data/working-ils.xlsx'
FEATURE_SET_CHOICE = 'both'  # Options: "functional_groups", "molecular_descriptors", "both"
OVERRIDE_FEATURES = True
MANUAL_FEATURE_LIST = [
    'cation_CN', 'anion_LogP', 'anion_Molecular Surface Area', 'cation_TPSA',
    'cation_CH2', 'anion_Molecular Volume', 'cation_Molecular Weight',
    'anion_CH2', 'anion_Rotatable Bonds', 'anion_Br', 'cation_Molecular Volume',
    'cation_LogP', 'cation_aNCH2', 'cation_H-Bond Acceptors', 'anion_COOH',
    'anion_Molecular Radius', 'anion_TPSA', 'cation_Molecular Surface Area'
]
NUM_RUNS = 500
NUM_FEATURES = 35

# Step 1: Load and preprocess data
df = load_data(DATA_PATH)
included_data, excluded_data = preprocess_data(df)

# Step 2: Select features
X_included = select_features(included_data, FEATURE_SET_CHOICE, OVERRIDE_FEATURES, MANUAL_FEATURE_LIST)
y_included = included_data['Reference Viscosity']

# Step 3: Train-test split
X_train, X_test, y_train, y_test = split_data(X_included, y_included)

# Step 4: Train model
model = train_model(X_train, y_train)

# Step 5: Evaluate model
y_pred, r2_rand = evaluate_model(model, X_test, y_test)

# Step 6: Plot results
plot_results(included_data, excluded_data, y_test, y_pred, r2_rand)

# # Step 7: Feature importance
# feature_importance_df = calculate_feature_importance(model, X_included, NUM_FEATURES)
# plot_feature_importance(feature_importance_df, NUM_FEATURES)

# # Step 8: Confidence interval
# mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(X_included, y_included, NUM_RUNS)
# plot_confidence_interval(r2_scores, confidence_interval, mean_r2)

# # Print results
# print(f"Model R² on test data: {r2_rand:.2f}")
# print(f"95% Confidence Interval for R²: {confidence_interval}")
