import pandas as pd
import json
from itertools import combinations

# Load the json data from the file
with open('generate-fg-graph/ionic_liquid_info.json', 'r') as file:
    ionic_liquids_data = json.load(file)

# Function to calculate edge weight based on shared functional groups
def calculate_weight(groups1, groups2):
    shared_groups = set(groups1.keys()).intersection(set(groups2.keys()))
    weight = 0
    for group in shared_groups:
        weight += min(groups1[group], groups2[group])
    return weight

# List to store edges
edges = []

# Iterate over all pairs of ionic liquids
for idx, (il1, il2) in enumerate(combinations(ionic_liquids_data, 2)):

    il1_id = il1["IL ID"]
    il2_id = il2["IL ID"]


    # Compare cation functional groups
    cation_weight = calculate_weight(il1["Cation Functional Groups"], il2["Cation Functional Groups"])

    # Compare anion functional groups
    anion_weight = calculate_weight(il1["Anion Functional Groups"], il2["Anion Functional Groups"])

    # Total weight (if there is any shared group)
    total_weight = cation_weight + anion_weight

    if total_weight > 0:
        # Add an edge if there is at least one shared functional group
        edges.append({"Source": il1_id, "Target": il2_id, "Weight": total_weight})

# Convert the list of edges to a DataFrame
edges_df = pd.DataFrame(edges)

# Save to CSV
edges_csv_path = 'generate-fg-graph/gephi/edges.csv'
edges_df.to_csv(edges_csv_path, index=False)
