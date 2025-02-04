import pandas as pd

# Paths for input and output files
raw_dataset_path = 'data/xlsx/raw.xlsx'
data_path = 'data/xlsx/ils.xlsx'
output_path = 'data/xlsx/lightweight-ils_v2.xlsx'

# Load sheet with included functional groups
reference_model = pd.read_excel(raw_dataset_path, sheet_name="S9 | Reference model - SWMLR", header=1)
included_groups = reference_model.loc[reference_model["In model"] == True, "Group"].dropna().astype(str).tolist()

# Load the main dataset
df = pd.read_excel(data_path)

# Automatically identify columns to keep
general_columns = ["IL ID", "Cation", "Anion", "Excluded IL", "T / K", "η / mPa s"]

# Locate the index positions for cation molecular descriptor range
cation_start = df.columns.get_loc("cation_MaxAbsEStateIndex")
cation_end = df.columns.get_loc("cation_Molecular Radius") + 1
cation_mol_des_cols = df.columns[cation_start:cation_end].tolist()

# Locate the index positions for anion molecular descriptor range
anion_start = df.columns.get_loc("anion_MaxAbsEStateIndex")
anion_end = df.columns.get_loc("anion_Molecular Radius") + 1
anion_mol_des_cols = df.columns[anion_start:anion_end].tolist()

functional_group_cols = [col for col in df.columns if col.split('_')[-1] in included_groups]

# Combine all the columns to keep
cols_to_keep = general_columns + cation_mol_des_cols + anion_mol_des_cols + functional_group_cols

# Filter and clean the DataFrame
df = df[cols_to_keep].dropna()

# Keep only one row per "IL ID"
df = df.drop_duplicates(subset='IL ID', keep='first')

# Save the cleaned dataset to a new Excel file
df.to_excel(output_path, index=False)

print(f"Processed data saved to {output_path}")
