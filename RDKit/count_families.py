import pandas as pd

def count_unique_and_highest(file_path):
    # Load the Excel file
    data = pd.read_excel(file_path)

    # Count unique strings in 'cation_Family' and 'anion_Family'
    cation_counts = data['cation_Family'].value_counts()
    anion_counts = data['anion_Family'].value_counts()

    # Get the number of unique groups
    num_unique_cations = cation_counts.size
    num_unique_anions = anion_counts.size

    # Get the groups with the highest counts
    highest_cations = cation_counts.head()
    highest_anions = anion_counts.head()

    # Print results
    print(f"Number of unique cation families: {num_unique_cations}")
    print(f"Number of unique anion families: {num_unique_anions}")
    print("\nTop cation families:")
    print(highest_cations)
    print("\nTop anion families:")
    print(highest_anions)

# Example usage
file_path = 'RDKit/data/lightweight-ils.xlsx'
count_unique_and_highest(file_path)

# Number of unique cation families: 24
# Number of unique anion families: 15

# Top cation families:
# cation_Family
# imidazolium      715
# ammonium         399
# phosphonium      216
# pyridinium       156
# pyrrolidinium    123
# Name: count, dtype: int64

# Top anion families:
# anion_Family
# NTf2 derivatives    684
# carboxylates        205
# BF4 derivatives     183
# sulfonates          148
# inorganics          142
# Name: count, dtype: int64
