import streamlit as st
from constants import FEATURE_PRESET_OPTIONS
from upload_dataset import load_uploaded_data

def select_target(df):
    target = st.sidebar.selectbox("Select Target Variable", df.columns, placeholder="Select Target Variable", index=len(df.columns)-1)
    return target

def select_features(df, feature_selection_method):

    if feature_selection_method == "Use Feature Preset":
        return select_features_using_preset(df)

    elif feature_selection_method == "Use Feature Importance File":
        return select_features_using_file(df)

    elif feature_selection_method == "Manually Select Features":
        return select_features_manually(df)

    return None

def select_features_manually(df):
    selected_features = st.sidebar.multiselect("Manually Select Features", df.columns.tolist(), placeholder="Select Features")
    return selected_features

def select_features_using_file(df):
    feature_importance_file = st.sidebar.file_uploader("Upload Feature Importance File (Excel)", type=["xlsx"])
    if feature_importance_file:
        feature_importance_data = load_uploaded_data(feature_importance_file)
        selected_features = list(feature_importance_data['Feature'])
        select_features = df[select_features]
        return selected_features

def select_features_using_preset(df):
    selected_preset = st.sidebar.selectbox("Select Feature Set", list(FEATURE_PRESET_OPTIONS.values()), index=2)
    preset_key = next(key for key, value in FEATURE_PRESET_OPTIONS.items() if value == selected_preset)

    # Define functional group and molecular descriptor columns
    functional_group_cols = df.loc[:, "cation_Im13":"anion_cycNCH2"].columns

    # working-ils
    molecular_descriptor_cols = df.loc[:, "cation_Charge":"anion_Molecular Radius"].columns

    # working-ils_v2
    # molecular_descriptor_cols = df.loc[:, "cation_MaxAbsEStateIndex":"anion_Molecular Radius"].columns

    # Select features based on the user's choice
    if preset_key == "functional_groups":
        selected_features = functional_group_cols
    elif preset_key == "molecular_descriptors":
        selected_features = molecular_descriptor_cols
    elif preset_key == "both":
        selected_features = functional_group_cols.union(molecular_descriptor_cols)
    return selected_features
