import json
import csv
import pandas as pd

# Load the json data from the file
with open('generate-fg-graph/ionic_liquid_info.json', 'r') as file:
    ionic_liquids_data = json.load(file)  # Use json.load instead of json.read

# Create a list of nodes with their ID and Label
nodes = []
for ionic_liquid in ionic_liquids_data:
    node_id = ionic_liquid["IL ID"]
    label = f'{ionic_liquid["Cation"]} + {ionic_liquid["Anion"]}'
    nodes.append({"IL ID": node_id, "Label": label})

# Convert the list of nodes to a DataFrame
nodes_df = pd.DataFrame(nodes)

# Save to CSV
nodes_csv_path = 'generate-fg-graph/gephi/nodes.csv'
nodes_df.to_csv(nodes_csv_path, index=False, quoting=csv.QUOTE_ALL) # QUOTE_ALL essnetial for csv reader to interpret correctly
