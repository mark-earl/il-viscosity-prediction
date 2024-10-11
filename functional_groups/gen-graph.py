import pickle
import os
import pandas as pd
import csv
from tqdm import tqdm
import os

def generate_nodes_and_edges(df_database, df_ions):

    # File to cache the IL list
    cache_file = 'functional_groups/data/il_list_cache.pkl'

    # Check if the cached file exists
    if os.path.exists(cache_file):
        # Load IL list from cache
        print("Loading IL list from cache...")
        with open(cache_file, 'rb') as f:
            il_list = pickle.load(f)
    else:
        # Initialize an empty list to store IL entries
        il_list = []

        # Get the functional group columns (everything after 'Number of ILs composed of the ion')
        functional_group_columns = df_ions.columns[df_ions.columns.get_loc('Number of ILs composed of the ion') + 1:]

        # Iterate over each row in the database of ILs (df_database) with a progress bar
        for _, row in tqdm(df_database.iterrows(), total=df_database.shape[0], desc="Creating list of ILs and their functional groups"):
            cation = row['Cation']
            anion = row['Anion']

            # Initialize dictionaries to hold functional groups for this IL
            cation_groups = {}
            anion_groups = {}
            shared_groups = {}

            # Check functional groups for the cation
            cation_row = df_ions[df_ions['Abbreviation'] == cation]
            if not cation_row.empty:
                for group in functional_group_columns:
                    group_value = cation_row.iloc[0][group]
                    if group_value > 0:  # Check if the functional group is present (> 0)
                        cation_groups[group] = group_value  # Add group and value

            # Check functional groups for the anion
            anion_row = df_ions[df_ions['Abbreviation'] == anion]
            if not anion_row.empty:
                for group in functional_group_columns:
                    group_value = anion_row.iloc[0][group]
                    if group_value > 0:  # Check if the functional group is present (> 0)
                        anion_groups[group] = group_value  # Add group and value

            # Find shared groups between cation and anion
            for group in cation_groups:
                if group in anion_groups:
                    # Shared groups take the minimum value between the cation and anion group counts
                    shared_groups[group] = min(cation_groups[group], anion_groups[group])

            # Structure the result as per your example
            il_entry = {
                "IL": f"cation: {cation} + anion: {anion}",
                f"{cation} groups": [{"name": group, "value": value} for group, value in cation_groups.items()],
                f"{anion} groups": [{"name": group, "value": value} for group, value in anion_groups.items()],
                "shared functional groups": [{"name": group, "value": value} for group, value in shared_groups.items()]
            }

            # Append this IL entry to the list
            il_list.append(il_entry)

        # Save the IL list to cache for future use
        print("Saving IL list to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(il_list, f)

    # Create the nodes CSV file ('functional_groups/data/nodes.csv')
    with open('functional_groups/data/nodes.csv', 'w', newline='') as node_file:
        writer = csv.writer(node_file)
        writer.writerow(['ID', 'Label'])  # Write the header
        for idx, il_entry in enumerate(il_list):
            writer.writerow([idx, il_entry['IL']])  # Write each IL's index and label

    # Create the edges CSV file ('functional_groups/data/edges.csv')
    with open('functional_groups/data/edges.csv', 'w', newline='') as edge_file:
        writer = csv.writer(edge_file)
        writer.writerow(['Source', 'Target', 'Weight'])  # Write the header

        # Iterate through all pairs of ILs
        for i in tqdm(range(len(il_list)), desc="Creating edges"):
            for j in range(i + 1, len(il_list)):
                # Get the shared functional groups of both ILs
                il1_shared = set(group['name'] for group in il_list[i]['shared functional groups'])
                il2_shared = set(group['name'] for group in il_list[j]['shared functional groups'])

                # Find the intersection of shared functional groups
                common_groups = il1_shared.intersection(il2_shared)
                weight = len(common_groups)  # The weight is the number of shared groups

                if weight > 0:  # Only write edges if there are shared groups
                    writer.writerow([i, j, weight])  # Write source, target, and weight

def main():

    data_path = 'data/working_dataset.xlsx'

    # Load datasets
    df_ions = pd.read_excel(data_path, sheet_name='S2 | Ions')
    df_database = pd.read_excel(data_path, sheet_name='S3 | Database')

    # Generate nodes and edges
    generate_nodes_and_edges(df_database, df_ions)

if __name__ == "__main__":
    main()
