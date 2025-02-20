import streamlit as st
import pandas as pd
import os

@st.cache_data
def load_uploaded_data(data_file):
    return pd.read_excel(data_file) if data_file else None

def load_and_preview_dataset():
    st.sidebar.header("Step 1: Upload Dataset")
    data_file = st.sidebar.file_uploader("Upload Dataset (Excel)", type=["xlsx"])
    if data_file:
        df = load_uploaded_data(data_file)
        st.header("Dataset Preview")
        st.dataframe(df)
        return df

    else:
        df = load_uploaded_data(os.path.join(os.getcwd(), 'data', 'xlsx', 'working-ils.xlsx'))
        st.header("Dataset Preview")
        st.dataframe(df)
        return df
