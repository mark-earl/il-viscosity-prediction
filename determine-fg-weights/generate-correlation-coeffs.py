# Using Padsyknu S9, so no need

import pandas as pd

# Step 1: Load the dataset
file_path = "data/working-dataset-v3.xlsx"
df = pd.read_excel(file_path)

# Step 2a and 2b: Calculate viscosity values ("n")
n_values = []
for index, row in df.iterrows():
    a = row['a']
    b = row['b']
    try:
        # Solving for "n" using the given equation: (1/n) ** 0.3 = a + b(298)
        equation_value = a + b * 298
        n = (1 / equation_value) ** (1 / 0.3)
        n_values.append(n)
    except Exception:
        # If the calculation fails, append None
        n_values.append(None)

# Step 3a and 3b: Create binary vectors for columns after "Viscosity Group"
binary_columns = df.columns[df.columns.get_loc("Viscosity Group") + 1:]
# Create binary vectors with proper handling for complex numbers
binary_vectors = df[binary_columns].map(
    lambda x: 1 if isinstance(x, (int, float)) and x > 1 else 0
)

# Add values from n_values to the 'Viscosity_n' column
for i, value in enumerate(n_values):
    df.at[i, 'Viscosity_n'] = value

# Step 4a: Create a new DataFrame for output
output_columns = ['Dataset ID', 'IL ID', 'Cation', 'Anion', 'a', 'b', 'Viscosity_n']
binary_vector_columns = binary_vectors.columns
final_df = pd.concat([df[output_columns], binary_vectors], axis=1)

# Step 4b: Save to a new file
output_file_path = "data/correlation-dataset-v1.xlsx"
final_df.to_excel(output_file_path, index=False)
