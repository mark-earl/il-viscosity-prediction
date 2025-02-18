import streamlit as st
from upload_dataset import *
from constants import *
from data_preprocessing import *
from feature_selection import *
from model_training import model_training_step
from data_analysis import data_analysis_step

def show_features(X_included):
    st.header("Selected Features")
    features_list = X_included.columns.tolist()
    with st.expander(f"View Selected Features ({len(features_list)})"):
        for i in features_list:
            st.markdown("- " + i)

def main():

    st.title("Ionic Liquid Viscosity Prediction")
    st.sidebar.title("Configuration")

    # Let the user upload a dataset
    df = load_and_preview_dataset()

    if df is None:
        st.warning("No dataset uploaded.")
        return

    # Partition data based on 'Excluded IL' feature
    included_data, excluded_data = preprocess_data(df)

    # Remove irrelevant featrues from each data set
    included_data, excluded_data = remove_irrelevant_features(included_data, excluded_data)

    # Select features and target variable
    st.sidebar.header("Step 2: Select Features")
    feature_selection_method = st.sidebar.radio("Feature Selection Method",["Use Feature Preset","Use Feature Importance File", "Manually Select Features"])
    selected_features = select_features(included_data, feature_selection_method)

    target_feature = select_target(included_data)

    if len(selected_features) == 0:
        st.warning("No features selected. Please choose a preset, upload importance file, or select manually.")
        return

    X_included = included_data[selected_features]
    y_included = included_data[target_feature]

    show_features(X_included)

    st.sidebar.header("Step 3: Choose Desired Output")
    output = st.sidebar.radio("Output",["Run Machine Learning", "Perform Data Analysis"])

    if output == "Run Machine Learning":
        model_training_step(X_included, y_included, included_data, excluded_data)

    elif output == "Perform Data Analysis":
        data_analysis_step(X_included, y_included, included_data, excluded_data, selected_features)

if __name__ == "__main__":
    main()
