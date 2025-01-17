import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

run_confidence_interval = True
NUM_RUNS = 500
check_feature_importance = False

override_features = True
NUM_FEATURES = 35

# Optionally manually select features
manual_feature_list = [
    'cation_CN',
    'anion_LogP',
    'anion_Molecular Surface Area',
    'cation_TPSA',
    'cation_CH2',
    'anion_Molecular Volume',
    'cation_Molecular Weight',
    'anion_CH2',
    'anion_Rotatable Bonds',
    'anion_Br',
    'cation_Molecular Volume',
    'cation_LogP',
    'cation_aNCH2',
    'cation_H-Bond Acceptors',
    'anion_COOH',
    'anion_Molecular Radius',
    'anion_TPSA',
    'cation_Molecular Surface Area',
    # 'cation_Molecular Radius',
    # 'cation_CH', 'cation_Rotatable Bonds',
    # 'anion_Molecular Weight',
    # 'anion_Charge',
    # 'anion_(CH2)2PO4',
    # 'anion_CH2SO4'
]

# Load the data from the Excel file
data_path = 'RDKit/data/working-ils.xlsx'
df = pd.read_excel(data_path)

# Drop unnecessary columns
excluded_il_col = df["Excluded IL"]
df = df.drop(columns=[
    'IL ID', 'Cation', 'Anion'
])

# Filter data based on the "Excluded IL" attribute
included_data = df[excluded_il_col == False]
excluded_data = df[excluded_il_col == True]

# Define the columns for functional groups and molecular descriptors
functional_group_cols = df.loc[:, "cation_Im13":"anion_cycNCH2"].columns
molecular_descriptor_cols = df.loc[:, "cation_Charge":"anion_Molecular Radius"].columns

# Parameter to select feature set: "functional_groups", "molecular_descriptors", or "both"
feature_set_choice = "both"  # Change this to "functional_groups" or "molecular_descriptors" as needed

# Choose features based on selection
if feature_set_choice == "functional_groups":
    selected_features = functional_group_cols
elif feature_set_choice == "molecular_descriptors":
    selected_features = molecular_descriptor_cols
elif feature_set_choice == "both":
    selected_features = functional_group_cols.union(molecular_descriptor_cols)

# Subset data for selected features
X_included = included_data[selected_features]

# Filter X_included to use only manually selected features if list is provided
if override_features:
    X_included = X_included[manual_feature_list]

y_included = included_data['Reference Viscosity']  # Target

# Train-test split for included data
X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=0)

# Initialize and train the CatBoost model
model = CatBoostRegressor(
    # iterations=1000,
    # learning_rate=0.1,
    # depth=6,
    verbose=0
)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
r2_rand = r2_score(y_test, y_pred)

# Prepare data for plotting
included_log_viscosities = np.log10(included_data['Reference Viscosity'])
excluded_log_viscosities = np.log10(excluded_data['Reference Viscosity'])
y_test_log = np.log10(y_test)
y_pred_log = np.log10(y_pred)

# Plot
plt.figure(figsize=(12, 8))

# Plot excluded data (gray)
plt.scatter(
    np.log10(excluded_data['Reference Viscosity']),
    np.log10(excluded_data['Reference Viscosity']),
    alpha=0.6, color='gray', label="Excluded ILs"
)

# Plot included data (blue)
plt.scatter(
    included_log_viscosities,
    included_log_viscosities,
    alpha=0.6, color='blue', label="Included ILs"
)

# Plot test data (red)
plt.scatter(
    y_pred_log, y_test_log,
    alpha=0.8, color='red', label=f"Test Data (R²: {r2_rand:.2f})"
)

# Add ideal fit line
plt.plot(
    [included_log_viscosities.min(), included_log_viscosities.max()],
    [included_log_viscosities.min(), included_log_viscosities.max()],
    'r--', lw=2, label="Ideal Fit"
)

# Customize plot
plt.title("Log-Scale Predicted vs Actual Viscosities")
plt.xlabel("Predicted Log Viscosity (log10[mPa s])")
plt.ylabel("Actual Log Viscosity (log10[mPa s])")
plt.legend()
plt.tight_layout()
plt.show()

# FEATURE IMPORTANCE
if check_feature_importance:

    # Assuming X_included and the trained CatBoost model are already available

    # Step 1: Retrieve Feature Importances
    feature_importances = model.get_feature_importance()
    feature_names = X_included.columns

    # Combine names and importances into a DataFrame for better readability
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Print the top NUM_FEATURES features
    print("Top NUM_FEATURES Features by Importance:")
    print(feature_importance_df.head(NUM_FEATURES))

    # Step 2: Visualize Feature Importances
    plt.figure(figsize=(12, 8))
    plt.barh(
        feature_importance_df['Feature'].head(NUM_FEATURES)[::-1],  # Reverse order for horizontal bar plot
        feature_importance_df['Importance'].head(NUM_FEATURES)[::-1],
        color='skyblue'
    )
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Top {NUM_FEATURES} Important Features')
    plt.tight_layout()
    plt.show()

# CONFIDENCE INTERVAL CALCULATIONS
if run_confidence_interval:

    # Run the model NUM_RUNS times and collect R² values
    r2_scores = []

    for i in tqdm(range(NUM_RUNS), desc="Training and Evaluating"):
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i)

        # Train the CatBoost model
        model = CatBoostRegressor(
            # iterations=1000,
            # learning_rate=0.1,
            # depth=6,
            verbose=0  # Suppress output for multiple runs
        )
        model.fit(X_train, y_train)

        # Predict and evaluate
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    # Calculate 95% confidence interval for R²
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)  # Sample standard deviation
    confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(NUM_RUNS))

    # Print confidence interval
    print(f"95% Confidence Interval for R²: {confidence_interval}")

    # Sort R² scores for a smooth line plot
    plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")

    # Highlight the confidence interval
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3, label="95% CI")

    # Plot mean R² as a horizontal line
    plt.axhline(mean_r2, color='green', linestyle='-', label="Mean R²")

    # Labels and title
    plt.title(f"R² Scores Across {NUM_RUNS} Runs with 95% Confidence Interval")
    plt.xlabel("Run Index")
    plt.ylabel("R² Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
