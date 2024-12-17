from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from rdkit.Chem.AllChem import UFFOptimizeMolecule, EmbedMolecule
import pandas as pd
import rdkit.Chem.rdFreeSASA as FreeSASA  # For surface area calculation

# Load dataset
dataset_path = 'data/raw.xlsx'
output_path = 'RDKit/data/ions.xlsx'

# Read the original ions data
df_ions = pd.read_excel(dataset_path, sheet_name='S2 | Ions')

# Drop rows where the SMILES is empty
df_ions = df_ions.dropna(subset=["SMILES"]) # No SMILES for hydrogenebisfluoride

# Function to compute properties for a SMILES string
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        print("Error - mol not derived from SMILES correctly")

    properties = {
        "Molecular Weight": Descriptors.MolWt(mol),  # g/mol
        "LogP": Descriptors.MolLogP(mol),            # LogP
        "TPSA": Descriptors.TPSA(mol),               # Topological polar surface area (Å²)
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }
    return properties

# Prepare the new properties columns
property_columns = [
    "Molecular Weight", "LogP", "TPSA", "H-Bond Donors",
    "H-Bond Acceptors", "Rotatable Bonds"
]

# Initialize the columns with NaN
for col in property_columns:
    df_ions[col] = None

# Reorder columns to place the new properties after the "M / g/mol" column
cols = df_ions.columns.tolist()
molecular_weight_index = cols.index("M / g/mol")
new_order = (cols[:molecular_weight_index + 1] + property_columns + cols[molecular_weight_index + 1:-6])
df_ions = df_ions[new_order]

# Compute properties for each ion and update the DataFrame
for idx, row in df_ions.iterrows():
    smiles = row["SMILES"]

    properties = compute_properties(smiles)

    if properties:
        for col, value in properties.items():
            df_ions.at[idx, col] = value

# Export the final DataFrame to a new Excel file
df_ions.to_excel(output_path, index=False, sheet_name='Computed Ions')

print(f"Processed data saved to {output_path}")
