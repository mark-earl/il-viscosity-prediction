import pandas as pd
import json

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

def select_features(df, feature_set_choice, override_features, feature_json_path):
    # Define functional group and molecular descriptor columns
    functional_group_cols = df.loc[:, "cation_Im13":"anion_cycNCH2"].columns
    molecular_descriptor_cols = df.loc[:, "cation_Charge":"anion_Molecular Radius"].columns

    # Select features based on the user's choice
    if feature_set_choice == "functional_groups":
        selected_features = functional_group_cols
    elif feature_set_choice == "molecular_descriptors":
        selected_features = molecular_descriptor_cols
    elif feature_set_choice == "both":
        selected_features = functional_group_cols.union(molecular_descriptor_cols)

    X_included = df[selected_features]

    # Override features if specified, using a JSON file
    if override_features and feature_json_path:
        with open(feature_json_path, 'r') as f:
            feature_data = json.load(f)  # Load feature data from JSON
        # Extract feature names from the "Feature" key
        manual_feature_list = [item["Feature"] for item in feature_data]
        X_included = X_included[manual_feature_list]


    return X_included
