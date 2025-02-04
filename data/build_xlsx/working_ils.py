import pandas as pd
import numpy as np

# Load the specified sheets from the provided Excel files
raw_data_path = 'data/xlsx/raw_write.xlsx'
lightweight_data_path = 'data/xlsx/lightweight-ils.xlsx'

# Load raw data sheet "S7 | Modeling - correction term"
raw_data = pd.read_excel(raw_data_path, sheet_name='S7 | Modeling - correction term')

# Remove duplicates in raw data based on "Cation" and "Anion"
raw_data_deduplicated = raw_data.drop_duplicates(subset=['Cation', 'Anion'])

# Load RDKit data
rdkit_data = pd.read_excel(lightweight_data_path)

# Perform the matching based on "Cation" and "Anion"
merged_data = pd.merge(
    rdkit_data,
    raw_data_deduplicated[['Cation', 'Anion', "η0 /mPa s"]],
    on=['Cation', 'Anion'],
    how='left'
)

# Add the matched "η0 /mPa s" values as a new column "Reference Viscosity"
merged_data.rename(columns={'η0 /mPa s': 'Reference Viscosity'}, inplace=True)

# Perform log transformation on the Reference Viscosity column and handle non-positive values
merged_data['Reference Viscosity Log'] = np.log(merged_data['Reference Viscosity'].replace(0, np.nan)).fillna(0)

# Remove experimental temp and viscosity
merged_data = merged_data.drop(columns=['T / K', 'η / mPa s'])

# Reorder columns to move "cation_Family" and "anion_Family" after "Cation" and "Anion"
columns_order = ['Cation', 'Anion', 'cation_Family', 'anion_Family'] + [
    col for col in merged_data.columns if col not in ['Cation', 'Anion', 'cation_Family', 'anion_Family']
]
merged_data = merged_data[columns_order]

# Save the updated RDKit file with the new column order
updated_rdkit_path = 'data/xlsx/working-ils.xlsx'
merged_data.to_excel(updated_rdkit_path, index=False)
