import pandas as pd

# Load the data from the Excel file
def load_data(data_path):
    return pd.read_excel(data_path)

# Drop unnecessary columns
def preprocess_data(df):
    excluded_il_col = df["Excluded IL"]
    df = df.drop(columns=[
        'IL ID', 'Cation', 'Anion'
    ])

    included_data = df[excluded_il_col == False]
    excluded_data = df[excluded_il_col == True]

    return included_data, excluded_data

# Define the columns for functional groups and molecular descriptors
def select_features(df, feature_set_choice, override_features, manual_feature_list):
    functional_group_cols = df.loc[:, "cation_Im13":"anion_cycNCH2"].columns
    molecular_descriptor_cols = df.loc[:, "cation_Charge":"anion_Molecular Radius"].columns

    if feature_set_choice == "functional_groups":
        selected_features = functional_group_cols
    elif feature_set_choice == "molecular_descriptors":
        selected_features = molecular_descriptor_cols
    elif feature_set_choice == "both":
        selected_features = functional_group_cols.union(molecular_descriptor_cols)

    X_included = df[selected_features]

    if override_features:
        X_included = X_included[manual_feature_list]

    return X_included
