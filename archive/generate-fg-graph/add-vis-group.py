import pandas as pd

# File path
xlsx_path = 'data/working-dataset-v2.xlsx'
output_path = 'data/working-dataset-v3.xlsx'

data = pd.read_excel(xlsx_path)
viscosity_column = 'η / mPa s'
viscosity_median = data[viscosity_column].median()  # Median of viscosity values

# Assign groups "H" for high and "L" for low
data['Viscosity Group'] = data[viscosity_column].apply(
    lambda x: 'H' if x >= viscosity_median else 'L'
)

cols = list(data.columns)
b_index = cols.index('b')  # Find index of 'b' column
im_index = cols.index('Im')  # Find index of 'Im' column


cols.remove('Viscosity Group')  # Ensure it's not already in the columns
cols.insert(b_index + 1, 'Viscosity Group')

data = data[cols]
data.to_excel(output_path, index=False)
print(f"Output saved as {output_path}")
