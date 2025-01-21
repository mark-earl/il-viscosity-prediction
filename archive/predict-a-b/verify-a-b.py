import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Load the dataset
dataset_path = 'data/working-dataset-v3.xlsx'
data = pd.read_excel(dataset_path)

# Select required columns: 'a', 'b', 'T / K', and 'η / mPa s'
data = data[['a', 'b', 'T / K', 'η / mPa s']].dropna()

# Function to remove outliers using the Interquartile Range (IQR) method
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    dropped_count = len(df) - len(filtered_df)
    return filtered_df, dropped_count

# Calculate the viscosity 'n' (calculated)
def calculate_n(row):
    a = row['a']
    b = row['b']
    T = row['T / K']
    calculated_n = 1 / ((a + b * T) ** (1 / .3))
    return calculated_n

# Apply the formula to calculate viscosity
data['calculated_n'] = data.apply(calculate_n, axis=1)

# Compute the residuals between experimental and calculated viscosities
data['residual'] = np.abs(data['η / mPa s'] - data['calculated_n'])

# Define a threshold for the residuals (e.g., 20% of the calculated viscosity)
threshold = 0.2 * data['calculated_n']

# Remove rows where the residual exceeds the threshold
data_cleaned_pred_outliers = data[data['residual'] <= threshold]

# Track how many rows were dropped
dropped_count_pred_outliers = len(data) - len(data_cleaned_pred_outliers)
print(f"Number of nodes dropped as predicted outliers: {dropped_count_pred_outliers}")

# Plot Experimental vs Calculated Viscosity after removing predicted outliers
plt.figure(figsize=(8, 6))
plt.scatter(data_cleaned_pred_outliers['η / mPa s'], data_cleaned_pred_outliers['calculated_n'], color='blue', alpha=0.7)
plt.plot([data_cleaned_pred_outliers['η / mPa s'].min(), data_cleaned_pred_outliers['η / mPa s'].max()],
         [data_cleaned_pred_outliers['η / mPa s'].min(), data_cleaned_pred_outliers['η / mPa s'].max()],
         color='red', linestyle='--', label='Ideal Fit (y=x)')
plt.xlabel('Experimental Viscosity (η / mPa s)')
plt.ylabel('Calculated Viscosity (n)')
plt.title('Experimental vs Calculated Viscosity (After Removing Predicted Outliers)')
plt.legend()
plt.grid()
plt.show()

# Calculate R^2 between experimental and calculated viscosity after removing predicted outliers
r2_pred_outliers = r2_score(data_cleaned_pred_outliers['η / mPa s'], data_cleaned_pred_outliers['calculated_n'])
print(f"R^2 between Experimental and Calculated Viscosity (after removing predicted outliers): {r2_pred_outliers:.4f}")

# Create a second graph showing values < 3500 on both axes
data_filtered = data_cleaned_pred_outliers[(data_cleaned_pred_outliers['η / mPa s'] < 3500) & (data_cleaned_pred_outliers['calculated_n'] < 3500)]

# Plot Experimental vs Calculated Viscosity with values < 3500
plt.figure(figsize=(8, 6))
plt.scatter(data_filtered['η / mPa s'], data_filtered['calculated_n'], color='green', alpha=0.7)
plt.plot([data_filtered['η / mPa s'].min(), data_filtered['η / mPa s'].max()],
         [data_filtered['η / mPa s'].min(), data_filtered['η / mPa s'].max()],
         color='red', linestyle='--', label='Ideal Fit (y=x)')
plt.xlabel('Experimental Viscosity (η / mPa s)')
plt.ylabel('Calculated Viscosity (n)')
plt.title('Experimental vs Calculated Viscosity (Filtered < 3500)')
plt.legend()
plt.grid()
plt.show()
