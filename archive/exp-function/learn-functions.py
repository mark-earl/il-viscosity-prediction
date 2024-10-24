import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from numpy.polynomial import Polynomial

# Use: v_i = a_i * e ^ (b_i / RT)
# OR log(v_i) = log(a_i) + (b_i / RT)
# OR log(v_i) = c_0 + c_1 (1/t)
# u_n = 1 / t_n
# w_n = log(v_n)
# something in scikit learn will be able to determine c_0 and c_1
# c_0 = log(a_i) and c_1 = b` (b/R)
# w_i = c_0 + c_1*u_i
# At least 2 ILs, leave out ILs with only 1 entry
# Test MSE & R^2
# Have a and b reported in the dataset

# Load Excel file
excel_path = 'data/raw.xlsx'  # Adjust path as needed

# Load the specific sheets
df = pd.read_excel(excel_path, sheet_name='S8 | Modeling vs "raw" database')
ions_df = pd.read_excel(excel_path, sheet_name='S2 | Ions')
groups_df = pd.read_excel(excel_path, sheet_name='S1 | Groups')

# Identify the start of functional group columns in S2 | Ions
start_col = 'Number of ILs composed of the ion'  # The last known column before functional groups
start_idx = ions_df.columns.get_loc(start_col) + 1  # Get the index of the first functional group column
functional_group_cols = ions_df.columns[start_idx:]  # Extract all functional group columns

# Group data based on 'IL ID'
grouped = df.groupby('IL ID')

# Helper function to calculate chemical similarity between two ILs based on shared functional groups
def calculate_similarity(ion1, ion2):
    # Extract functional groups for both ions
    ion1_groups = ion1[functional_group_cols].values.flatten()
    ion2_groups = ion2[functional_group_cols].values.flatten()

    # Ensure both arrays have the same length, else return 0 similarity
    if len(ion1_groups) != len(ion2_groups):
        return 0

    # Calculate similarity by counting matching functional groups
    return np.sum(ion1_groups == ion2_groups)

# Find the most similar IL group based on functional groups if R² is below threshold
def find_most_similar_ion(il_id):
    current_ion = ions_df[ions_df['Abbreviation'] == il_id]
    if current_ion.empty:
        return None  # If no ion found, return None

    other_ions = ions_df[ions_df['Abbreviation'] != il_id]

    # Calculate similarity scores with other ions
    similarity_scores = other_ions.apply(lambda row: calculate_similarity(current_ion.iloc[0], row), axis=1)

    # If no matches found, return None
    if similarity_scores.empty:
        return None

    # Return the IL with the highest similarity score
    most_similar_ion = other_ions.iloc[similarity_scores.idxmax()]
    return most_similar_ion['Abbreviation']

# Function to fit the best model and evaluate
def process_group(group):
    # Separate features (T/K) and target (η / mPa s)
    X = group[['T / K']].values.flatten()  # input feature
    y = group['η / mPa s'].values  # target variable

    # Split the group into train/test if the number of rows is more than 1
    if len(group) > 1:
        # Case 1: More than 3 samples -> use Linear Regression
        if len(group) > 3:
            X_train, X_test, y_train, y_test = train_test_split(X.reshape(-1, 1), y, test_size=0.2, random_state=42)
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = r2_score(y_test, y_pred)  # Use test set for R2 score

        # Case 2: 2-3 samples -> use Polynomial fit
        elif 2 <= len(group) <= 3:
            # Polynomial degree 2
            p = Polynomial.fit(X, y, deg=2)
            y_pred = p(X)  # Predict on the same X values
            accuracy = r2_score(y, y_pred)  # Use full set for R2 score

        # If the accuracy is below the threshold (0.80), find the most similar IL and use its prediction
        if accuracy < 0.80:
            similar_il = find_most_similar_ion(group['IL ID'].iloc[0])
            if similar_il:
                print(f"R² < 0.80 for {group['IL ID'].iloc[0]}. Using most similar IL: {similar_il}")

                # Use the most similar IL's data for prediction
                similar_group = df[df['IL ID'] == similar_il]
                X_similar = similar_group[['T / K']].values.flatten()
                y_similar = similar_group['η / mPa s'].values
                if len(X_similar) > 1:  # Ensure the similar IL has enough data
                    X_train, X_test, y_train, y_test = train_test_split(X_similar.reshape(-1, 1), y_similar, test_size=0.2, random_state=42)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    accuracy = r2_score(y_test, y_pred)
            else:
                print(f"No similar IL found for {group['IL ID'].iloc[0]}.")

        return accuracy

    else:
        return None  # Not enough data to split into train/test

# Iterate through each group and process
results = {}
for name, group in grouped:
    accuracy = process_group(group)
    if accuracy is not None:
        results[name] = accuracy

# Convert the results to a DataFrame and sort by accuracy
results_df = pd.DataFrame(list(results.items()), columns=['IL ID', 'R^2 Accuracy'])
results_df = results_df.sort_values(by='R^2 Accuracy', ascending=False)

# Save the sorted results to a CSV file
output_csv_path = 'exp-function/sorted_results.csv'
results_df.to_csv(output_csv_path, index=False)

# Calculate the average R² accuracy
average_r2 = results_df['R^2 Accuracy'].mean()

# Print the average R² accuracy
print(f"Average R² accuracy: {average_r2:.4f}")

results_df.head(), output_csv_path
