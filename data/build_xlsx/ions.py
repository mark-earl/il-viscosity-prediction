from rdkit import Chem
from rdkit.Chem import Descriptors
import rdkit.Chem.rdFreeSASA as FreeSASA
from rdkit.Chem.AllChem import UFFOptimizeMolecule, EmbedMolecule
import pandas as pd
import math
from tqdm import tqdm

# Load datasets
dataset_path = 'data/xlsx/raw.xlsx'
output_path = 'data/xlsx/ions.xlsx'

# Read the original ions data and drop rows where the SMILES is empty
df_ions = pd.read_excel(dataset_path, sheet_name='S2 | Ions').dropna(subset=["SMILES"])

# Compute molecular properties for a given SMILES
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
    except ValueError:
        print(f"Skipping ion due to conformer generation/optimization error: {smiles}")
        return None

    # Compute all RDKit molecular descriptors
    properties = {}
    for name, func in Descriptors.descList:
        try:
            properties[name] = func(mol)
        except Exception:
            properties[name] = None

    # Compute molecular surface area, volume, and radius
    try:
        radii = FreeSASA.classifyAtoms(mol_h)
        sasa = FreeSASA.CalcSASA(mol_h, radii)
        properties.update({
            "Molecular Surface Area": round(sasa, 3),
            "Molecular Volume": round(sasa / 100, 3),
            "Molecular Radius": round((3 * (sasa / 100) / (4 * math.pi)) ** (1/3), 3),
        })
    except Exception:
        properties.update({
            "Molecular Surface Area": None,
            "Molecular Volume": None,
            "Molecular Radius": None,
        })

    return properties


# Compute descriptors and add them to DataFrame
all_descriptors = [name for name, _ in Descriptors.descList]
df_ions = df_ions.reindex(columns=df_ions.columns.tolist() + all_descriptors + ["Molecular Surface Area", "Molecular Volume", "Molecular Radius"])

skipped_ions = []
for idx, row in tqdm(df_ions.iterrows(), total=len(df_ions), desc="Computing descriptors"):
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
