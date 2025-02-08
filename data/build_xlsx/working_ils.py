import pandas as pd
import numpy as np

# Define file paths
raw_data_path = 'data/xlsx/raw_write.xlsx'
lightweight_data_path = 'data/xlsx/lightweight-ils_v2.xlsx'
updated_rdkit_path = 'data/xlsx/working-ils_v2.xlsx'

# Load raw data and remove duplicates based on "Cation" and "Anion"
raw_data = pd.read_excel(raw_data_path, sheet_name='S7 | Modeling - correction term')
raw_data = raw_data.drop_duplicates(subset=['Cation', 'Anion'])

# Load RDKit data
rdkit_data = pd.read_excel(lightweight_data_path)

# Merge RDKit data with deduplicated raw data based on "Cation" and "Anion"
merged_data = pd.merge(
    rdkit_data,
    raw_data[['Cation', 'Anion', 'η0 /mPa s']],
    on=['Cation', 'Anion'],
    how='left'
)

# Add "Reference Viscosity" and compute log transformation
merged_data.rename(columns={'η0 /mPa s': 'Reference Viscosity'}, inplace=True)
merged_data['Reference Viscosity Log'] = np.log(merged_data['Reference Viscosity'].replace(0, np.nan)).fillna(0)

# Remove experimental temperature and viscosity columns if they exist
merged_data.drop(columns=['T / K', 'η / mPa s'], errors='ignore', inplace=True)

# Save the updated RDKit file with the new column order
merged_data.to_excel(updated_rdkit_path, index=False)

print(f"Processed data saved to {updated_rdkit_path}")
