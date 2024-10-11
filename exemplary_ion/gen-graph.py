import pandas as pd
import os
from tqdm import tqdm

def generate_nodes_and_edges(df_database, df_groups, output_dir='exemplary_ion/data'):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Nodes: Create a CSV for ionic liquids from the database
    df_nodes = df_database[['Cation', 'Anion']].drop_duplicates().reset_index(drop=True)
    df_nodes['ID'] = df_nodes.index  # Unique ID for each IL
    df_nodes['Label'] = df_nodes['Cation'].astype(str) + " + " + df_nodes['Anion'].astype(str)  # Convert to strings and concatenate
    df_nodes[['ID', 'Label']].to_csv(os.path.join(output_dir, 'nodes.csv'), index=False)

    # Build a dictionary to map each IL to its functional groups
    # First, ensure there is a link between ILs and groups (if 'Exemplary ion' refers to either cation or anion)
    il_groups = {}

    # Adding a progress bar for the df_groups iteration
    for _, row in tqdm(df_groups.iterrows(), total=df_groups.shape[0], desc="Processing functional groups"):
        group_code = row['Group code']
        exemplary_ion = row['Exemplary ion']

        # Add the group to the list of groups for the corresponding IL
        if exemplary_ion in df_database['Cation'].values:
            il_groups.setdefault(exemplary_ion, []).append(group_code)
        elif exemplary_ion in df_database['Anion'].values:
            il_groups.setdefault(exemplary_ion, []).append(group_code)

    # Create a list to store edges
    edges = []

    # Loop through the ILs in the database and check for shared groups
    for i, il1 in tqdm(df_nodes.iterrows(), total=df_nodes.shape[0], desc="Processing edges between ILs (outer loop)"):
        groups_il1 = il_groups.get(il1['Cation'], []) + il_groups.get(il1['Anion'], [])

        for j, il2 in tqdm(df_nodes.iterrows(), total=df_nodes.shape[0], desc="Processing edges between ILs (inner loop)", leave=False):
            if i >= j:
                continue  # Avoid duplicate pairs and self-loops

            # Get the groups for the second IL
            groups_il2 = il_groups.get(il2['Cation'], []) + il_groups.get(il2['Anion'], [])

            # Find the shared groups
            shared_groups = set(groups_il1) & set(groups_il2)

            if shared_groups:
                edges.append([il1['ID'], il2['ID'], len(shared_groups)])  # Add edge with weight as the number of shared groups

    # Convert to DataFrame and save as CSV
    df_edges = pd.DataFrame(edges, columns=['Source', 'Target', 'Weight'])
    df_edges.to_csv(os.path.join(output_dir, 'edges.csv'), index=False)

    print(f"Generated nodes.csv and edges.csv in the {output_dir} folder.")

def main():
    # Load datasets
    df_groups = pd.read_excel(data_path, sheet_name='S1 | Groups')
    df_database = pd.read_excel(data_path, sheet_name='S3 | Database')

    # Generate nodes and edges
    generate_nodes_and_edges(df_database, df_groups)

data_path = 'C:/Users/mpe09/Development/il-property-prediction/data/working_dataset.xlsx'

if __name__ == "__main__":
    main()
