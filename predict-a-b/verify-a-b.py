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

# Remove outliers from 'η / mPa s' and track count
data_cleaned, dropped_nodes = remove_outliers(data, 'η / mPa s')
print(f"Number of nodes dropped as outliers: {dropped_nodes}")

# Calculate the viscosity 'n' (calculated)
def calculate_n(row):
    a = row['a']
    b = row['b']
    T = row['T / K']
    calculated_n = 1 / ((a + b * T) ** (1 /.3))
    return calculated_n

# Apply the formula to calculate viscosity
data_cleaned['calculated_n'] = data_cleaned.apply(calculate_n, axis=1)

# Plot Experimental vs Calculated Viscosity
plt.figure(figsize=(8, 6))
plt.scatter(data_cleaned['η / mPa s'], data_cleaned['calculated_n'], color='blue', alpha=0.7)
plt.plot([data_cleaned['η / mPa s'].min(), data_cleaned['η / mPa s'].max()],
         [data_cleaned['η / mPa s'].min(), data_cleaned['η / mPa s'].max()],
         color='red', linestyle='--', label='Ideal Fit (y=x)')
plt.xlabel('Experimental Viscosity (η / mPa s)')
plt.ylabel('Calculated Viscosity (n)')
plt.title('Experimental vs Calculated Viscosity (After Removing Outliers)')
plt.legend()
plt.grid()
plt.show()

# Calculate R^2 between experimental and calculated viscosity
r2 = r2_score(data_cleaned['η / mPa s'], data_cleaned['calculated_n'])
print(f"R^2 between Experimental and Calculated Viscosity (after removing outliers): {r2:.4f}")
