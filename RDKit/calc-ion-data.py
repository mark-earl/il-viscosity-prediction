from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import rdkit.Chem.rdFreeSASA as FreeSASA
from rdkit.Chem.AllChem import UFFOptimizeMolecule, EmbedMolecule
import pandas as pd
import math

# Load datasets
dataset_path = 'data/raw.xlsx'
output_path = 'RDKit/data/ions_extended.xlsx'

# Read the original ions data
df_ions = pd.read_excel(dataset_path, sheet_name='S2 | Ions')

# Drop rows where the SMILES is empty
df_ions = df_ions.dropna(subset=["SMILES"])  # No SMILES for hydrogenebisfluoride

# Compute relevant molecular properties for the various ions
def compute_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Error - mol not derived correctly for SMILES: {smiles}")
        return None

    # Generate 3D coordinates for geometry-based properties
    mol_h = Chem.AddHs(mol)
    try:
        EmbedMolecule(mol_h)
        UFFOptimizeMolecule(mol_h)
    except ValueError as e:
        print(f"Skipping ion due to conformer generation/optimization error: {smiles}. Error: {e}")
        return None

    # RDKit-based properties
    properties = {
        "Molecular Weight": Descriptors.MolWt(mol),  # g/mol
        "LogP": Descriptors.MolLogP(mol),           # LogP
        "TPSA": Descriptors.TPSA(mol),              # Topological polar surface area (Å²)
        "H-Bond Donors": rdMolDescriptors.CalcNumHBD(mol),
        "H-Bond Acceptors": rdMolDescriptors.CalcNumHBA(mol),
        "Rotatable Bonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
    }

    # Molecular surface area
    try:
        radii = FreeSASA.classifyAtoms(mol_h)
        sasa = FreeSASA.CalcSASA(mol_h, radii)  # Solvent accessible surface area
        properties["Molecular Surface Area"] = round(sasa, 3)

        # Approximate molecular volume using SASA (divided by a constant factor)
        volume = sasa / 100
        properties["Molecular Volume"] = round(volume, 3)

        # Molecular radius approximation
        radius = (3 * volume / (4 * math.pi)) ** (1/3)
        properties["Molecular Radius"] = round(radius, 3)
    except Exception as e:
        print(f"Skipping surface/volume calculations for SMILES {smiles}. Error: {e}")
        properties["Molecular Surface Area"] = None
        properties["Molecular Volume"] = None
        properties["Molecular Radius"] = None

    # Placeholder values for properties requiring quantum mechanics or external tools
    # TODO: possibly we could do this later?
    properties["Dipole Moment"] = "Requires QM tools"
    properties["Solvation-Free Energy"] = "Requires QM tools"
    properties["Misfit Interaction Energy"] = "Requires QM tools"
    properties["Sigma Moments"] = "To be calculated with QM tools"
    properties["Hydrogen Bond Interaction Energy"] = "Requires molecular dynamics"
    properties["Van der Waals Interaction Energy"] = "Requires molecular dynamics"
    properties["Total Mean Interaction Energy"] = "Requires QM calculations"

    return properties


# Prepare the new properties columns
property_columns = [
    "Molecular Weight", "LogP", "TPSA", "H-Bond Donors", "H-Bond Acceptors",
    "Rotatable Bonds", "Molecular Surface Area", "Molecular Volume",
    "Molecular Radius",

    # TODO: Relates to the previous categories
    # "Dipole Moment", "Solvation-Free Energy",
    # "Misfit Interaction Energy", "Sigma Moments", "Hydrogen Bond Interaction Energy",
    # "Van der Waals Interaction Energy", "Total Mean Interaction Energy"
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
skipped_ions = []
for idx, row in df_ions.iterrows():
    smiles = row["SMILES"]
    properties = compute_properties(smiles)
    if properties is None:
        skipped_ions.append(smiles)
    else:
        for col, value in properties.items():
            df_ions.at[idx, col] = value

print(f"Skipped the following ions due to errors: {skipped_ions}")

# Export the final DataFrame to a new Excel file
df_ions.to_excel(output_path, index=False, sheet_name='Extended Ions')

print(f"Processed data saved to {output_path}")
