import pandas as pd

raw_dataset_path = 'data/raw.xlsx'

# Load sheet with included functional groups
reference_model = pd.read_excel(raw_dataset_path, sheet_name="S9 | Reference model - SWMLR", header=1)
included_groups = reference_model[reference_model["In model"] == True]["Group"]

# Ensure the functional groups are valid column names
included_groups = included_groups.dropna().astype(str).tolist()

data_path = 'RDKit/data/ils.xlsx'
df = pd.read_excel(data_path)

# Define the columns to keep (given list)
cols_to_keep = [
    'IL ID', 'Cation', 'Anion', 'T / K', 'η / mPa s',
    'cation_Ion type', 'cation_Charge', 'cation_Family', 'cation_Molecular Weight', 'cation_LogP', 'cation_TPSA',
    'cation_H-Bond Donors', 'cation_H-Bond Acceptors', 'cation_Rotatable Bonds', 'cation_Number of ILs composed of the ion',
    'anion_Ion type', 'anion_Charge', 'anion_Family', 'anion_Molecular Weight', 'anion_LogP', 'anion_TPSA',
    'anion_H-Bond Donors', 'anion_H-Bond Acceptors', 'anion_Rotatable Bonds', 'anion_Number of ILs composed of the ion'
]

fg_cols = [col for col in df.columns if col.split('_')[-1] in included_groups]

cols_to_keep = cols_to_keep + fg_cols

df = df[cols_to_keep]

df = df.dropna()

# Save the cleaned dataset to a new Excel file
output_path = 'RDKit/data/heavyweight-ils.xlsx'
df.to_excel(output_path, index=False)

print(f"Processed data saved to {output_path}")
