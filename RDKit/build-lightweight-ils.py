import pandas as pd

# Step 1: Load the data from the Excel file
data_path = 'RDKit/data/ils.xlsx'
df = pd.read_excel(data_path)

# Step 2: Preprocess the data
# Drop irrelevant columns: 'Cation', 'Anion', 'cation_Ion type', 'anion_Ion type', 'cation_Family', 'anion_Family'
df = df.drop(columns=['Cation', 'Anion', 'cation_Ion type', 'anion_Ion type', 'cation_Family', 'anion_Family'])

# Step 3: Handle missing values (if necessary)
# Drop rows with missing values
df = df.dropna()

# Step 4: Keep only one row per "IL ID"
# Assuming 'IL ID' is a unique identifier for each ionic liquid, we'll take the first row for each unique 'IL ID'
df_unique = df.drop_duplicates(subset='IL ID', keep='first')

# Save the cleaned dataset to a new Excel file
output_path = 'RDKit/data/lightweight-ils.xlsx'
df_unique.to_excel(output_path, index=False)

print(f"Processed data saved to {output_path}")
