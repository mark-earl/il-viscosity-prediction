import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

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
included_data = df[df['Excluded IL'] == False]
excluded_data = df[df['Excluded IL'] == True]

# Apply log10 transformation to the target variable
included_data['Log_Reference_Viscosity'] = np.log10(included_data['Reference Viscosity'])
excluded_data['Log_Reference_Viscosity'] = np.log10(excluded_data['Reference Viscosity'])

# Define features and target variable for included data
X_included = included_data.drop(columns=['Reference Viscosity', 'Excluded IL', 'Log_Reference_Viscosity'])  # Features
y_included = included_data['Log_Reference_Viscosity']  # Log-transformed Target

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
y_pred_log = model.predict(X_test)

# Calculate R^2 for log-transformed data
r2_rand_log = r2_score(y_test, y_pred_log)

# Reverse-transform predictions for visualization
y_pred = 10**y_pred_log
y_test_actual = 10**y_test

# Prepare data for plotting
included_log_viscosities = np.log10(included_data['Reference Viscosity'])
excluded_log_viscosities = np.log10(excluded_data['Reference Viscosity'])

# Plot
plt.figure(figsize=(12, 8))

# Plot excluded data (gray)
plt.scatter(
    excluded_log_viscosities,
    excluded_log_viscosities,
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
    y_pred_log, y_test,
    alpha=0.8, color='red', label=f"Test Data (Log R²: {r2_rand_log:.2f})"
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
