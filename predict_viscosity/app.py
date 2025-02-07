import streamlit as st
import pandas as pd
from data_preprocessing import load_data, preprocess_data, select_features
from model_training import train_model, split_data
import json
from visualization import plot_results
from committee_confidence_interval import run_single_committee_model
from confidence_interval import calculate_confidence_interval, plot_confidence_interval

# Streamlit App Title
st.title("Ionic Liquid Viscosity Prediction")

# Sidebar Configuration
st.sidebar.title("Configuration")

st.sidebar.header("Step 1: Upload Dataset")

# File Input for Data
data_file = st.sidebar.file_uploader("Upload Dataset (Excel)", type=["xlsx"])

@st.cache_data
def load_data(data_file):
    df = pd.read_excel(data_file)
    return df

if data_file:

    # Load the uploaded dataset
    df = load_data(data_file)

    st.header("Dataset Preview:")
    st.dataframe(df)

    st.sidebar.header("Step 2: Select Features")

    # Checkbox for feature preset selection
    use_preset_checkbox = st.sidebar.checkbox("Use Feature Preset")

    selected_features = []
    preset_key = None
    manually_select_features = False
    feature_importance_file = None
    data_ready = False

    if use_preset_checkbox:
        # Dropdown for feature presets
        feature_preset_options = {
            "functional_groups": "Functional Groups",
            "molecular_descriptors": "Molecular Descriptors",
            "both": "Both"
        }

        # Display dropdown with readable names
        selected_preset = st.sidebar.selectbox("Select Feature Set", list(feature_preset_options.values()), index=2)

        # Map back to the corresponding key
        preset_key = next(key for key, value in feature_preset_options.items() if value == selected_preset)

    else:
        # Use Feature Importance File Option
        use_importance_file_checkbox = st.sidebar.checkbox("Use Feature Importance File")
        if use_importance_file_checkbox:
            # File upload for feature importance
            feature_importance_file = st.sidebar.file_uploader("Upload Feature Importance File (JSON)", type=["json"])
            if feature_importance_file:
                feature_importance_data = json.load(feature_importance_file)
                selected_features = [item["Feature"] for item in feature_importance_data]
                st.write("Selected Features from Importance File:", selected_features)
        else:
            # Manual feature selection
            manually_select_features = True
            selected_features = st.sidebar.multiselect("Manually Select Features", df.columns.tolist(),placeholder="Select Features")

    # Step 1: Preprocess data
    included_data, excluded_data = preprocess_data(df)

    # Step 2: Select features based on user input or feature importance file
    if preset_key or selected_features:

        manually_selected_features = None
        if manually_select_features:
            manually_selected_features = selected_features

        X_included = select_features(included_data, preset_key, manually_select_features, manually_selected_features, feature_importance_file)
        y_included = included_data["Reference Viscosity Log"]  # Assuming this is your target column
        data_ready = True
        st.header("Features Selected for Model Training:")
        st.write(X_included.columns.tolist())
    else:
        st.warning("No features selected. Please choose preset, upload importance file, or select manually.")


    # Placeholder for Model Training
    if data_ready:

        st.sidebar.header("Step 3: Choose Desired Output")

        output = st.sidebar.radio("Output",["Run Machine Learning", "Perform Data Analysis"])


        if output == "Perform Data Analysis":
            st.sidebar.header("Step 4: Choose Data Analysis Option")

        elif output == "Run Machine Learning":

            MODELS = {
                "catboost" : "CatBoost",
                "xgboost" : "XGBoost",
                "random_forest" : "Random Forest",
                "lightgbm" : "LightGBM",
                "gradient_boosting" : "Gradient Boosting",
                "adaboost" : "ADABoost",
                "linear_regression" : "Linear Regression",
                "ridge" : "Ridge",
                "lasso" : "Lasso",
                "elastic_net" : "Elastic Net",
                "svr" : "Support Vector Regression",
                "knn" : "K-Nearest-Neighbors",
                "decision_tree" : "Decision Trees",
                "mlp" : "Multi-Layer Perceptron"
            }

            st.sidebar.header("Step 4: Select Model")

            use_comittee = False
            if st.sidebar.checkbox("Use Committee"):
                use_comittee = True
                committees = st.sidebar.multiselect("Select Committee Models", MODELS.values(),placeholder="Select Models")

            else:
                model_name = st.sidebar.selectbox("Select Model", list(MODELS.values()))
                model_key = next(key for key, value in MODELS.items() if value == model_name)

            run_ci = False
            if st.sidebar.checkbox("Generate Confidence Interval"):
                run_ci = True
                num_runs = st.sidebar.slider("Number of Runs", min_value=0, max_value=250, value=1, step=5)

            st.sidebar.header("Step 5: Train Model")

            if st.sidebar.button("Train Model"):

                if run_ci:
                    mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(X_included, y_included, num_runs, model_key)

                    if use_comittee:
                        st.write(f"Commite of: {', '.join(committees)} trained successfully!")
                    else:
                        st.write(f"{model_name} trained successfully!")

                    st.pyplot(plot_confidence_interval(r2_scores, confidence_interval, mean_r2))

                elif not X_included.empty:
                    # Train-test split
                    X_train, X_test, y_train, y_test = split_data(X_included, y_included)

                    if use_comittee:
                        y_pred, r2_rand = run_single_committee_model(X_train, X_test, y_train, y_test)
                        model_name = f"Commite of: {', '.join(committees)}"

                    else:
                        model = train_model(X_train, y_train, model_key)
                        y_pred, r2_rand = model.predict(X_test), model.score(X_test, y_test)

                    st.write(f"{model_name} trained successfully!")

                    # Evaluate the model
                    st.header("R² Score on Test Data:")
                    s = f"<p style='font-size:40px;color:#0096FF;'>{r2_rand:.3f}</p>"
                    st.markdown(s, unsafe_allow_html=True)

                    st.pyplot(plot_results(included_data, excluded_data, y_test, y_pred, r2_rand))
                else:
                    st.error("No valid features selected for training.")
