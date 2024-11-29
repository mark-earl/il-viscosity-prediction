import pandas as pd

# File paths
xlsx_path = 'data/working-dataset-v3.xlsx'
csv_path = 'generate-fg-graph/gephi/nodesv2.csv'
output_path = 'generate-fg-graph/gephi/nodesv3.csv'

# Step 1: Load the Excel file and CSV file
xlsx_data = pd.read_excel(xlsx_path)
csv_data = pd.read_csv(csv_path)

# Step 2: Merge data based on "IL ID" to match rows from the xlsx file with the csv
merged_data = csv_data.merge(
    xlsx_data[['IL ID', 'T / K']],  # Only need IL ID and Viscosity Group columns
    on='IL ID',                             # Join on 'IL ID'
    how='left'                              # Left join to preserve all rows from the csv
)

# Step 3: Save the updated CSV with the new "Viscosity Group" column
merged_data.to_csv(output_path, index=False)

print(f"Output saved as {output_path}")
