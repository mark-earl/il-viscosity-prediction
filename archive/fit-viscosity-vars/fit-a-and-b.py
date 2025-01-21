# Import necessary libraries
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
from tqdm import tqdm

# Load the dataset
file_path = 'data/working-dataset.xlsx'
df = pd.read_excel(file_path)

# Group data by 'IL ID' and filter groups with < 2 values
grouped = df.groupby('IL ID').filter(lambda x: len(x) > 1)

# Prepare to store results
results = []

# Loop over each group, fit the equation, and compute R^2
for il_id, group in tqdm(grouped.groupby('IL ID'), desc="Processing IL ID groups"):
    T = group['T / K'].values
    viscosity = group['η / mPa s'].values
    cation = group['Cation'].values[0]
    anion = group['Anion'].values[0]

    # If there are exactly 2 data points, use a simple linear fit
    if len(T) == 2:
        slope, intercept = np.polyfit(T, 1 / viscosity ** 0.3, 1)
        predicted = intercept + slope * T
        r2 = 1.0  # R² is 1 for two points perfectly fitting a line
        results.append({'IL ID': il_id, 'Cation': cation, 'Anion': anion, 'a': intercept, 'b': slope, 'R^2': r2})

    else:
        initial_guess = [1, 1]

        try:
            # Perform initial curve fitting
            popt, _ = curve_fit(lambda T, a, b: a + b * T, T, 1 / viscosity ** 0.3, p0=initial_guess)
            predicted = popt[0] + popt[1] * T

            # Calculate residuals and standard deviation
            residuals = 1 / viscosity ** 0.3 - predicted
            std_dev = np.std(residuals)

            # Filter out points that have residuals > 1.1 * std_dev
            inliers = np.abs(residuals) <= 1.1 * std_dev
            T_inliers = T[inliers]
            viscosity_inliers = viscosity[inliers]

            # Re-fit curve on filtered data
            popt, _ = curve_fit(lambda T, a, b: a + b * T, T_inliers, 1 / viscosity_inliers ** 0.3, p0=initial_guess)
            predicted_inliers = popt[0] + popt[1] * T_inliers
            r2 = r2_score(1 / viscosity_inliers ** 0.3, predicted_inliers)

            # Store results
            results.append({'IL ID': il_id, 'Cation': cation, 'Anion': anion, 'a': popt[0], 'b': popt[1], 'R^2': r2})

        except RuntimeError:
            results.append({'IL ID': il_id, 'Cation': cation, 'Anion': anion, 'a': None, 'b': None, 'R^2': None})

# Convert results to DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='R^2', ascending=False)

# Save results to CSV
output_file = 'fit-viscosity-vars/fitting_results_sorted.csv'
results_df.to_csv(output_file, index=False)
print(results_df.head())

# Compute and report the average R^2 score and number of ILs tested
valid_r2_df = results_df.dropna(subset=['R^2'])
average_r2 = valid_r2_df['R^2'].mean()
num_il_tested = len(valid_r2_df)
print(f"\nAverage R^2 score: {average_r2:.4f}")
print(f"Number of ionic liquids tested: {num_il_tested}")
