import pandas as pd
from tqdm import tqdm

# Load the first four columns from 'S8 | Modeling vs "raw" database' sheet
ils_data_path = 'data/raw.xlsx'
df_ils = pd.read_excel(ils_data_path, sheet_name='S8 | Modeling vs "raw" database', usecols=[1, 2, 3, 6, 8, 9])

# Drop any rows in the "df_ils" where either that "Cation" col or the "Anion" col is "hydrogenebisfluoride"
# Drop rows where "Cation" or "Anion" is "hydrogenebisfluoride"
df_ils_cleaned = df_ils[~df_ils["Cation"].str.lower().eq("hydrogenebisfluoride")]
df_ils_cleaned = df_ils_cleaned[~df_ils_cleaned["Anion"].str.lower().eq("hydrogenebisfluoride")]

df_ils = df_ils_cleaned

# Load ions data
ions_data_path = 'RDKit/data/ions_extended.xlsx'
columns_to_skip = ['Chemical name', 'SMILES', 'Faimly', 'M / g/mol', 'Dipole Moment', 'Solvation-Free Energy', 'Misfit Interaction Energy', 'Sigma Moments', 'Hydrogen Bond Interaction Energy', 'Van der Waals Interaction Energy', 'Total Mean Interaction Energy']
df_ions = pd.read_excel(ions_data_path, sheet_name='Extended Ions', usecols=lambda x: x not in columns_to_skip)

# Function to get ion properties and prefix by ion type
def get_ion_properties(ion_type, abbreviation):
    ion_row = df_ions[df_ions['Abbreviation'] == abbreviation]
    if ion_row.empty:
        print("WARNING - Ion row is empty")
        return {}

    # For each column in ion_row, prefix with the ion_type
    ion_properties = {}
    for col in ion_row.columns:
        # Skip Abbreviation since it is already used as part of the mapping
        if col == 'Abbreviation':
            continue
        ion_properties[f"{ion_type}_{col}"] = ion_row[col].values[0]
    return ion_properties

# Create an empty list to store the processed rows
processed_data = []

# Iterate over each row in df_ils
for idx, row in tqdm(df_ils.iterrows(), total=len(df_ils), desc="Processing rows", unit="row"):
    cation = row["Cation"]
    anion = row["Anion"]

    # Get properties for both cation and anion
    cation_properties = get_ion_properties("cation", cation)
    anion_properties = get_ion_properties("anion", anion)

    # Combine the original row with the cation and anion properties
    new_row = row.to_dict()
    new_row.update(cation_properties)
    new_row.update(anion_properties)

    # Append the new row to the processed data list
    processed_data.append(new_row)

# Convert the processed data into a DataFrame
df_processed = pd.DataFrame(processed_data)

output_path = 'RDKit/data/ils.xlsx'
df_processed.to_excel(output_path, index=False)
