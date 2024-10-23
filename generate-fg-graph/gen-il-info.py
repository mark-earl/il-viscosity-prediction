import pandas as pd
import json
import numpy as np

# Step 1: Open the Excel spreadsheet
file_path = "data/raw.xlsx"
xls = pd.ExcelFile(file_path)

# Step 2: Open the sheet "S3 | Database" and "S2 | Ions"
df_database = pd.read_excel(xls, sheet_name="S3 | Database")
df_ions = pd.read_excel(xls, sheet_name="S2 | Ions")

# Step 3: Separate the rows of Ionic Liquids by the "IL ID" column
ionic_liquids = df_database.groupby('IL ID')

# Step 4: Get the "Cation" and "Anion" attribute for each "IL ID"
ionic_liquid_info = []

# for il_id, il_group in ionic_liquids:
for il_id, il_group in ionic_liquids:

    cation = il_group['Cation'].values[0]
    anion = il_group['Anion'].values[0]

    # Step 5: Find the "Cation" and "Anion" in the "Abbreviation" column of the "S2 | Ions" sheet
    cation_row = df_ions[df_ions['Abbreviation'] == cation]
    anion_row = df_ions[df_ions['Abbreviation'] == anion]

    cation_groups = {}
    anion_groups = {}

    # Step 6a and 6b: Get functional groups for Cation and Anion
    if not cation_row.empty:
        for col in cation_row.columns[8:]: # 8 is the start of the functional groups in the "S2 | Ions sheet"
            if cation_row[col].values[0] > 0:
                cation_groups[col] = cation_row[col].values[0]

    if not anion_row.empty:
        for col in anion_row.columns[8:]:
            if anion_row[col].values[0] > 0:
                anion_groups[col] = anion_row[col].values[0]

    # Step 7: Add the number and name of the functional groups onto the cation/anion
    ionic_liquid_info.append({
        'IL ID': il_id,
        'Cation': cation,
        'Cation Functional Groups': cation_groups,
        'Anion': anion,
        'Anion Functional Groups': anion_groups
    })

# File path to export the JSON data
output_file_path = "generate-fg-graph/ionic_liquid_info.json"

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

# Step 7: Export the ionic liquid info to a JSON file
with open(output_file_path, 'w') as json_file:
    json.dump(ionic_liquid_info, json_file, indent=4, cls=NpEncoder)
