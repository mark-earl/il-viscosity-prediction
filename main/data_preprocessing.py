import streamlit as st
from constants import *

def remove_irrelevant_features(included_data, excluded_data):

    # Ensure included_data and excluded_data share the exact same cols
    if len(included_data.columns) != len(excluded_data.columns):
        st.warning("Included data and excluded data have different schema")
        return

    features_to_drop = st.sidebar.multiselect("Features removed for ML & data processing", included_data.columns, IRRELEVANT_FEATURES_WORKING_ILS)

    included_data = included_data.drop(features_to_drop, axis='columns')
    excluded_data = excluded_data.drop(features_to_drop, axis='columns')

    return included_data, excluded_data

@st.cache_data
def preprocess_data(df):
    excluded_il_col = df["Excluded IL"]

    included_data = df[excluded_il_col == False]
    excluded_data = df[excluded_il_col == True]

    return included_data, excluded_data
