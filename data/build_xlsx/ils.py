import pandas as pd
from tqdm import tqdm

# Paths for input and output files
ils_data_path = 'data/xlsx/raw.xlsx'
ions_data_path = 'data/xlsx/ions.xlsx'
output_path = 'data/xlsx/ils.xlsx'

# Load the relevant columns from the 'S8 | Modeling vs "raw" database' sheet
df_ils = pd.read_excel(ils_data_path, sheet_name='S8 | Modeling vs "raw" database', usecols=[1, 2, 3, 6, 8, 9])

# Remove rows where either "Cation" or "Anion" is "hydrogenebisfluoride"
df_ils = df_ils[
    (~df_ils["Cation"].str.lower().eq("hydrogenebisfluoride")) &
    (~df_ils["Anion"].str.lower().eq("hydrogenebisfluoride"))
]

# Load ions data, skipping unnecessary columns
columns_to_skip = [
    'Chemical name', 'SMILES', '**Family', 'M / g/mol', 'Dipole Moment',
    'Solvation-Free Energy', 'Misfit Interaction Energy', 'Sigma Moments',
    'Hydrogen Bond Interaction Energy', 'Van der Waals Interaction Energy',
    'Total Mean Interaction Energy'
]
df_ions = pd.read_excel(ions_data_path, sheet_name='Extended Ions', usecols=lambda x: x not in columns_to_skip)


def get_ion_properties(ion_type, abbreviation):
    """
    Fetches properties for a given ion type and abbreviation,
    prefixes the properties by the ion type, and returns them as a dictionary.
    """
    ion_row = df_ions[df_ions['Abbreviation'] == abbreviation]

    if ion_row.empty:
        print(f"WARNING - No matching ion found for {ion_type} '{abbreviation}'")
        return {}

    # Prefix column names with the ion type
    return {f"{ion_type}_{col}": ion_row[col].values[0] for col in ion_row.columns if col != 'Abbreviation'}


# Process each row in df_ils and enrich it with cation/anion properties
processed_data = []

for _, row in tqdm(df_ils.iterrows(), total=len(df_ils), desc="Processing rows", unit="row"):
    cation_abbr = row["Cation"]
    anion_abbr = row["Anion"]

    # Get properties for both cation and anion
    cation_properties = get_ion_properties("cation", cation_abbr)
    anion_properties = get_ion_properties("anion", anion_abbr)

    # Combine the row data with cation and anion properties
    enriched_row = row.to_dict()
    enriched_row.update(cation_properties)
    enriched_row.update(anion_properties)

    processed_data.append(enriched_row)

# Convert the processed data into a DataFrame
df_processed = pd.DataFrame(processed_data)

# Export the final DataFrame to an Excel file
df_processed.to_excel(output_path, index=False)

print(f"Processed data saved to {output_path}")
