import pandas as pd
import joblib
import os

def get_data(data_path, cache_path, sheet_names):
    if os.path.exists(cache_path):
        print("Loaded from cache")
        return joblib.load(cache_path)
    else:
        print("Reading from Excel and caching")
        dfs = pd.read_excel(data_path, sheet_name=sheet_names)
        joblib.dump(dfs, cache_path)
        return dfs

def save_data_to_excel(dfs, output_path):
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Data saved to {output_path}")

# Define paths and sheet names
data_path = 'data/raw.xlsx'
cache_path = 'data/cached_data.pkl'
sheet_names = ['S1 | Groups', 'S2 | Ions', 'S3 | Database']
output_path = 'data/working_dataset.xlsx'

# Get the data
dfs = get_data(data_path, cache_path, sheet_names)

# Save the data to a new Excel file
save_data_to_excel(dfs, output_path)

# Access your DataFrame
groups_df = dfs['S1 | Groups']
ions_df = dfs['S2 | Ions']
database_df = dfs['S3 | Database']
