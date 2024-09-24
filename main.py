import pandas as pd
import joblib
import os
import requests
import json

def get_data(data_path, cache_path, sheet_names):
    if os.path.exists(cache_path):
        print("Loaded from cache")
        return joblib.load(cache_path)
    else:
        print("Reading from Excel and caching")
        dfs = pd.read_excel(data_path, sheet_name=sheet_names)
        joblib.dump(dfs, cache_path)
        return dfs

def save_data_to_excel(dfs, output_path):
    # Check if file exists
    if os.path.exists(output_path):
        print(f"File {output_path} already exists.")
        # You can prompt the user, overwrite, or create a backup file here
        return

    # Save the data if the file doesn't exist
    with pd.ExcelWriter(output_path) as writer:
        for sheet_name, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data saved to {output_path}")

# Define paths and sheet names
data_path = 'data/raw.xlsx'
cache_path = 'data/cached_data.pkl'
sheet_names = ['S1 | Groups', 'S2 | Ions', 'S3 | Database']
output_path = 'data/working_dataset.xlsx'
not_found_cache_path = 'data/not_found_cache.json'

# Load the cache of SMILES strings that do not have associated images
if os.path.exists(not_found_cache_path):
    with open(not_found_cache_path, 'r') as f:
        not_found_cache = json.load(f)
else:
    not_found_cache = {}

# Get the data
dfs = get_data(data_path, cache_path, sheet_names)

# Save the data to a new Excel file
save_data_to_excel(dfs, output_path)

# Access your DataFrame
groups_df = dfs['S1 | Groups']
ions_df = dfs['S2 | Ions']
database_df = dfs['S3 | Database']

def get_pubchem_cid_from_smiles(smiles_string):
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{smiles_string}/cids/JSON"

    try:
        response = requests.get(url)
        data = response.json()

        if 'IdentifierList' in data and 'CID' in data['IdentifierList']:
            return data['IdentifierList']['CID'][0]

    except Exception as e:
        print(f'Error fetching CID for {smiles_string}: {e}')
        return None

def get_compound_image(cid, image_size='100x100'):
    # URL for fetching the image of the compound in PNG format
    image_url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/PNG?image_size={image_size}"
    return image_url

def save_image(image_url, file_path):
    response = requests.get(image_url)
    if response.status_code == 200:
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f"Image saved to {file_path}")
    else:
        print(f"Failed to retrieve image from {image_url}")

good = 0
total = 0

for smiles_string in ions_df['SMILES']:
    # Check if the SMILES string is empty
    if type(smiles_string) == float or not smiles_string.strip():  # This checks for empty strings or strings with only whitespace
        print("Encountered an empty SMILES string. Skipping.")
        continue  # Skip this iteration if the SMILES string is empty

    # Check if the SMILES string is in the not found cache
    if smiles_string in not_found_cache:
        print(f"Skipping {smiles_string}, cached result indicates no image found.")
        total += 1
        continue

    # Create a safe filename from the SMILES string
    safe_smiles = smiles_string.replace('/', '_').replace('\\', '_')  # Replace any problematic characters
    image_path = f"data/images/{safe_smiles}.png"

    # Check if the image already exists
    if os.path.exists(image_path):
        good += 1
        total += 1
        print(f"Image for {smiles_string} already exists.")
    else:
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

# Save the updated cache to file
with open(not_found_cache_path, 'w') as f:
    json.dump(not_found_cache, f)

if total > 0:
    print(f"Success rate: {good / total:.2%}")
else:
    print("No compounds processed.")
