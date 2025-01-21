import pandas as pd
import joblib
import os
import requests
import json

from rdkit import Chem
from rdkit.Chem import Draw

def load_data(excel_data_path, excel_cache_path, sheet_names):
    if os.path.exists(excel_cache_path):
        print("Loaded from cache")
        return joblib.load(excel_cache_path)

    print("Reading from Excel and caching")
    dfs = pd.read_excel(excel_data_path, sheet_name=sheet_names)
    joblib.dump(dfs, cache_path)
    return dfs

def save_data_to_excel(dfs, output_path):
    if os.path.exists(output_path):
        print(f"File {output_path} already exists.")
        return

    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"Data saved to {output_path}")

def load_not_found_cache(cache_path):
    if os.path.exists(cache_path):
        with open(cache_path, 'r') as f:
            return json.load(f)
    return {}

def save_not_found_cache(cache_path, not_found_cache):
    with open(cache_path, 'w') as f:
        json.dump(not_found_cache, f)

def get_pubchem_cid_from_smiles(smiles_string):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_string}/cids/JSON"
    try:
        response = requests.get(url)
        data = response.json()
        return data['IdentifierList']['CID'][0] if 'IdentifierList' in data and 'CID' in data['IdentifierList'] else None
    except Exception as e:
        print(f'Error fetching CID for {smiles_string}: {e}')
        return None

def get_compound_image(cid, image_size='100x100'):
    return f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size={image_size}"

def save_image(image_url, file_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved to {file_path}")
    else:
        print(f"Failed to retrieve image from {image_url}")

def draw_molecule(smiles_string, file_path, image_size=(100, 100)):
    """Draws a molecule from a SMILES string and saves it as a PNG."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol:
        img = Draw.MolToImage(mol, size=image_size)
        img.save(file_path)
        print(f"Image drawn and saved to {file_path}")
    else:
        print(f"Could not convert SMILES to molecule for: {smiles_string}")

def main(data_path, cache_path, output_path, not_found_cache_path):
    # Load datasets and caches
    dfs = load_data(data_path, cache_path, ['S1 | Groups', 'S2 | Ions', 'S3 | Database'])
    not_found_cache = load_not_found_cache(not_found_cache_path)

    # Save the data to a new Excel file
    save_data_to_excel(dfs, output_path)

    good, total = 0, 0

    # Draw molecules for SMILES in not_found_cache
    for smiles_string in not_found_cache.keys():
        safe_smiles = smiles_string.replace('/', '_').replace('\\', '_')
        image_path = f"data/images/{safe_smiles}.png"
        draw_molecule(smiles_string, image_path)

    # Process the 'S2 | Ions' DataFrame for images
    for smiles_string in dfs['S2 | Ions']['SMILES']:
        if isinstance(smiles_string, str) and smiles_string.strip():
            if smiles_string in not_found_cache:
                print(f"Skipping {smiles_string}, cached result indicates no image found.")
                total += 1
                continue

            safe_smiles = smiles_string.replace('/', '_').replace('\\', '_')
            image_path = f"data/images/{safe_smiles}.png"

            if os.path.exists(image_path):
                good += 1
                total += 1
                print(f"Image for {smiles_string} already exists.")
                continue

            cid = get_pubchem_cid_from_smiles(smiles_string)
            if cid:
                good += 1
                total += 1
                image_url = get_compound_image(cid)
                save_image(image_url, image_path)
            else:
                total += 1
                print(f"Compound '{smiles_string}' not found.")
                not_found_cache[smiles_string] = True  # Cache the not found SMILES string
        else:
            print("Encountered an empty or invalid SMILES string. Skipping.")
            total += 1

    # Save the updated cache to file
    save_not_found_cache(not_found_cache_path, not_found_cache)

    if total > 0:
        print(f"Success rate: {good / total:.2%}")
    else:
        print("No compounds processed.")

# Define paths
data_path = 'data/raw.xlsx'
cache_path = 'data/cached_data.pkl'
output_path = 'data/working_dataset.xlsx'
not_found_cache_path = 'data/not_found_cache.json'

if __name__ == "__main__":
    main(data_path, cache_path, output_path, not_found_cache_path)
