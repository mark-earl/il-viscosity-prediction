import pandas as pd

# Load the Excel file and CSV file
working_dataset_path = 'data/working-dataset.xlsx'
fitting_results_path = 'fit-viscosity-vars/fitting_results_sorted.csv'

# Load the working dataset and fitting results
working_df = pd.read_excel(working_dataset_path)
fitting_df = pd.read_csv(fitting_results_path)

# Drop duplicates to keep only the first entry for each "IL ID" in the working dataset
working_df = working_df.drop_duplicates(subset="IL ID", keep="first")

# Create a dictionary from the fitting results with "IL ID" as keys and (a, b) pairs as values
fitting_dict = fitting_df.set_index('IL ID')[['a', 'b']].to_dict(orient='index')

# Map "a" and "b" values to the working dataset using the dictionary
working_df['a'] = working_df['IL ID'].map(lambda x: fitting_dict.get(x, {}).get('a', None))
working_df['b'] = working_df['IL ID'].map(lambda x: fitting_dict.get(x, {}).get('b', None))

# Save the updated working dataset to a new Excel file
updated_path = 'data/updated-working-dataset.xlsx'
working_df.to_excel(updated_path, index=False)
