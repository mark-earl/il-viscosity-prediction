import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from tqdm import tqdm

run_confidence_interval = False

# Load the data from the Excel file
data_path = 'RDKit/data/working-ils.xlsx'
df = pd.read_excel(data_path)

# Drop unnecessary columns
excluded_il_col = df["Excluded IL"]
df = df.drop(columns=[
    'IL ID', 'Cation', 'Anion',
    'T / K', 'η / mPa s'
])

# Filter data based on the "Excluded IL" attribute
included_data = df[excluded_il_col == False]
excluded_data = df[excluded_il_col == True]

# Define features and target variable for included data
X_included = included_data.drop(columns=['Reference Viscosity', 'Excluded IL'])  # Features
y_included = included_data['Reference Viscosity']  # Target

# Train-test split for included data
X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=0)

# Initialize and train the CatBoost model
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=200
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

# CONFIDENCE INTERVAL CALCULATIONS

if not run_confidence_interval:
    quit()

# Run the model 50 times and collect R² values
num_runs = 50
r2_scores = []

for i in tqdm(range(num_runs), desc="Training and Evaluating"):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i)

    # Train the CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=6,
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
confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(num_runs))

# Print confidence interval
print(f"95% Confidence Interval for R²: {confidence_interval}")

# Sort R² scores for a smooth line plot
plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")

# Highlight the confidence interval
plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3, label="95% CI")

# Plot mean R² as a horizontal line
plt.axhline(mean_r2, color='green', linestyle='-', label="Mean R²")

# Labels and title
plt.title("R² Scores Across 50 Runs with 95% Confidence Interval")
plt.xlabel("Run Index")
plt.ylabel("R² Values")
plt.legend()
plt.tight_layout()
plt.show()
