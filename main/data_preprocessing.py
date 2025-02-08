def preprocess_data(df):
    excluded_il_col = df["Excluded IL"]
    # Drop unnecessary columns
    df = df.drop(columns=[
        'IL ID', 'Cation', 'Anion', 'Reference Viscosity'
    ])

    included_data = df[excluded_il_col == False]
    excluded_data = df[excluded_il_col == True]

    return included_data, excluded_data

def select_features(df, preset_key, manually_select_features, manually_selected_features, feature_json_path):
    # Define functional group and molecular descriptor columns
    functional_group_cols = df.loc[:, "cation_Im13":"anion_cycNCH2"].columns
    molecular_descriptor_cols = df.loc[:, "cation_Charge":"anion_Molecular Radius"].columns

    if preset_key:
        # Select features based on the user's choice
        if preset_key == "functional_groups":
            selected_features = functional_group_cols
        elif preset_key == "molecular_descriptors":
            selected_features = molecular_descriptor_cols
        elif preset_key == "both":
            selected_features = functional_group_cols.union(molecular_descriptor_cols)

        X_included = df[selected_features]
        return X_included

    # Override features if specified, using a JSON file
    if manually_select_features:
        X_included = df[manually_selected_features]
        return X_included

    return None
