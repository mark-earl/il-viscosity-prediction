import streamlit as st
import pandas as pd
import json
from data_preprocessing import preprocess_data, select_features
from model_training import train_model, split_data
from visualization import plot_results
from confidence_interval_utils import (
    calculate_confidence_interval,
    plot_confidence_interval,
    run_single_committee_model
)

# Define model display names
MODELS = {
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "ADABoost",
    "linear_regression": "Linear Regression",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "elastic_net": "Elastic Net",
    "svr": "Support Vector Regression",
    "knn": "K-Nearest-Neighbors",
    "decision_tree": "Decision Trees",
    "mlp": "Multi-Layer Perceptron"
}

FEATURE_PRESET_OPTIONS = {
    "functional_groups": "Functional Groups",
    "molecular_descriptors": "Molecular Descriptors",
    "both": "Both"
}

st.title("Ionic Liquid Viscosity Prediction")
st.sidebar.title("Configuration")


@st.cache_data
def load_uploaded_data(data_file):
    return pd.read_excel(data_file) if data_file else None


def load_and_preview_dataset():
    data_file = st.sidebar.file_uploader("Upload Dataset (Excel)", type=["xlsx"])
    if data_file:
        df = load_uploaded_data(data_file)
        st.header("Dataset Preview:")
        st.dataframe(df)
        return df
    return None


def select_features_step(df):
    st.sidebar.header("Step 2: Select Features")
    use_preset_checkbox = st.sidebar.checkbox("Use Feature Preset")

    if use_preset_checkbox:
        selected_preset = st.sidebar.selectbox("Select Feature Set", list(FEATURE_PRESET_OPTIONS.values()), index=2)
        preset_key = next(key for key, value in FEATURE_PRESET_OPTIONS.items() if value == selected_preset)
        return preset_key, None, None

    use_importance_file_checkbox = st.sidebar.checkbox("Use Feature Importance File")
    if use_importance_file_checkbox:
        feature_importance_file = st.sidebar.file_uploader("Upload Feature Importance File (JSON)", type=["json"])
        if feature_importance_file:
            feature_importance_data = json.load(feature_importance_file)
            selected_features = [item["Feature"] for item in feature_importance_data]
            st.write("Selected Features from Importance File:", selected_features)
            return None, selected_features, None

    selected_features = st.sidebar.multiselect("Manually Select Features", df.columns.tolist(), placeholder="Select Features")
    return None, selected_features, None


def model_training_step(X_included, y_included, included_data, excluded_data):
    st.sidebar.header("Step 4: Select Model")
    use_committee = st.sidebar.checkbox("Use Committee")

    if use_committee:
        committees = st.sidebar.multiselect("Select Committee Models", MODELS.values(), placeholder="Select Models")
        committee_keys = [key for key, value in MODELS.items() if value in committees]
    else:
        model_name = st.sidebar.selectbox("Select Model", list(MODELS.values()))
        model_key = next(key for key, value in MODELS.items() if value == model_name)

    run_ci = st.sidebar.checkbox("Generate Confidence Interval")
    num_runs = st.sidebar.slider("Number of Runs", min_value=1, max_value=250, value=1, step=5) if run_ci else 0

    st.sidebar.header("Step 5: Train Model")

    if st.sidebar.button("Train Model"):
        if run_ci:
            if use_committee:
                mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(
                    X_included, y_included, num_runs, committee_models=committee_keys
                )
                st.write(f"Committee of: {', '.join(committees)} trained successfully!")
                st.pyplot(plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix="(Committee)"))
            else:
                mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(
                    X_included, y_included, num_runs, model_name=model_key
                )
                st.write(f"{model_name} trained successfully!")
                st.pyplot(plot_confidence_interval(r2_scores, confidence_interval, mean_r2))
        else:
            X_train, X_test, y_train, y_test = split_data(X_included, y_included)
            if use_committee:
                y_pred, r2_rand = run_single_committee_model(X_train, X_test, y_train, y_test, committee_keys)
                st.write(f"Committee of: {', '.join(committees)} trained successfully!")
                st.pyplot(plot_results(included_data, excluded_data, y_test, y_pred, r2_rand))
            else:
                model = train_model(X_train, y_train, model_key)
                y_pred, r2_rand = model.predict(X_test), model.score(X_test, y_test)
                st.write(f"{model_name} trained successfully!")
                st.header("R² Score on Test Data:")
                st.markdown(f"<p style='font-size:40px;color:#0096FF;'>{r2_rand:.3f}</p>", unsafe_allow_html=True)
                st.pyplot(plot_results(included_data, excluded_data, y_test, y_pred, r2_rand))


def main():
    df = load_and_preview_dataset()

    if df is not None:
        included_data, excluded_data = preprocess_data(df)
        preset_key, selected_features, feature_importance_file = select_features_step(df)

        if preset_key or selected_features:
            manually_selected_features = selected_features if selected_features else None
            X_included = select_features(included_data, preset_key, bool(selected_features), manually_selected_features, feature_importance_file)
            y_included = included_data["Reference Viscosity Log"]  # Assuming this is the target column
            st.header("Features Selected for Model Training:")
            st.write(X_included.columns.tolist())
            model_training_step(X_included, y_included, included_data, excluded_data)
        else:
            st.warning("No features selected. Please choose a preset, upload importance file, or select manually.")


if __name__ == "__main__":
    main()
