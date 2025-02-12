import streamlit as st
import pandas as pd
from data_preprocessing import preprocess_data, select_features
from model_training import train_model, split_data
from visualization import (
    plot_results,
    plot_confidence_interval
)
from data_analysis import (
    plot_graph_relationships,
    plot_feature_importance,
    plot_correlation_heatmap
)
from confidence_interval_utils import (
    calculate_confidence_interval,
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

MODELS_WITH_FEATURE_IMPORTANCE = {
    "catboost": "CatBoost",
    "xgboost": "XGBoost",
    "random_forest": "Random Forest",
    "lightgbm": "LightGBM",
    "gradient_boosting": "Gradient Boosting",
    "adaboost": "ADABoost",
    "decision_tree": "Decision Trees",
}

FEATURE_PRESET_OPTIONS = {
    "functional_groups": "Functional Groups",
    "molecular_descriptors": "Molecular Descriptors",
    "both": "Both"
}

EXCLUDED_FEATURES = [
    'IL ID',
    'Cation',
    'Anion',
    'cation_Family',
    'anion_Family',
    'Excluded IL'
]

st.title("Ionic Liquid Viscosity Prediction")
st.sidebar.title("Configuration")


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
    return None

def data_analysis_step(X_included, y_included, included_data, excluded_data, selected_features):
    st.sidebar.header("Data Analysis Options")

    feature_importance = st.sidebar.checkbox("Analyze Feature Importance")
    if feature_importance:
        num_features = st.sidebar.slider("Number of features to display", min_value=1, max_value=len(X_included.columns), value=1)
        if num_features:
            committee_keys = model_key = None
            use_committee = st.sidebar.checkbox("Use Committee")
            if use_committee:
                committees = st.sidebar.multiselect("Select Committee Models", MODELS_WITH_FEATURE_IMPORTANCE.values(), placeholder="Select Models")
                committee_keys = [key for key, value in MODELS_WITH_FEATURE_IMPORTANCE.items() if value in committees]
            else:
                model_name = st.sidebar.selectbox("Select Model for Feature Importance Analysis", list(MODELS_WITH_FEATURE_IMPORTANCE.values()))
                model_key = next(key for key, value in MODELS_WITH_FEATURE_IMPORTANCE.items() if value == model_name)
            if st.sidebar.button("Analyze Features"):
                plot_feature_importance(X_included, y_included, num_features, use_committee, committee_keys, model_key)

    corr_heatmap = st.sidebar.checkbox("Generate Correlational Heatmap")
    if corr_heatmap:
        num_heatmap_features = st.sidebar.slider("Number of Features for Heatmap", min_value=2, max_value=len(X_included.columns), value=10)
        if st.sidebar.button("Generate Heatmap"):
            plot_correlation_heatmap(X_included, y_included, num_heatmap_features)

    graph = st.sidebar.checkbox("Generate Graph")
    if graph:
        if st.sidebar.button("Generate Graph"):
            if selected_features:
                plot_graph_relationships(X_included, selected_features)
            else:
                st.warning("Please select at least one feature for graph generation.")

def select_features_step(df):
    st.sidebar.header("Step 2: Select Features")
    feature_selection_method = st.sidebar.radio("Feature Selection Method",["Use Feature Preset","Use Feature Importance File", "Manually Select Features"])

    if feature_selection_method == "Use Feature Preset":
        selected_preset = st.sidebar.selectbox("Select Feature Set", list(FEATURE_PRESET_OPTIONS.values()), index=2)
        preset_key = next(key for key, value in FEATURE_PRESET_OPTIONS.items() if value == selected_preset)
        return preset_key, None, None

    elif feature_selection_method == "Use Feature Importance File":
        feature_importance_file = st.sidebar.file_uploader("Upload Feature Importance File (Excel)", type=["xlsx"])
        if feature_importance_file:
            feature_importance_data = load_uploaded_data(feature_importance_file)
            selected_features = list(feature_importance_data['Feature'])
            return None, selected_features, None

    elif feature_selection_method == "Manually Select Features":
        selected_features = st.sidebar.multiselect("Manually Select Features", df.columns.tolist(), placeholder="Select Features")
        return None, selected_features, None

    return None, None, None

def select_target_step(df):
    features = [feature for feature in df.columns.tolist() if feature not in EXCLUDED_FEATURES]
    target = st.sidebar.selectbox("Select Target Variable", features, placeholder="Select Target Variable", index=len(features)-1)
    return target

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
    num_runs = st.sidebar.slider("Number of Runs", min_value=0, max_value=250, value=1, step=5) if run_ci else 0

    st.sidebar.header("Step 5: Train Model")

    if st.sidebar.button("Train Model"):
        if run_ci:
            if use_committee:
                mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(
                    X_included, y_included, num_runs, committee_models=committee_keys
                )
                st.write(f"Committee of: {', '.join(committees)} trained successfully!")
            else:
                mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(
                    X_included, y_included, num_runs, model_name=model_key
                )
                st.write(f"{model_name} trained successfully!")

            plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix="(Committee)" if use_committee else "")

        else:
            X_train, X_test, y_train, y_test = split_data(X_included, y_included)
            if use_committee:
                y_train_pred, y_pred, r2_rand = run_single_committee_model(X_train, X_test, y_train, y_test, committee_keys)
            else:
                model = train_model(X_train, y_train, model_key)
                y_train_pred, y_pred = model.predict(X_train), model.predict(X_test)
                r2_rand = model.score(X_test, y_test)

            st.header("R² Score on Test Data")
            st.markdown(f"<p style='font-size:40px;color:#0096FF;'>{r2_rand:.3f}</p>", unsafe_allow_html=True)
            plot_results(y_train, y_train_pred, y_test, y_pred, r2_rand)

def main():
    df = load_and_preview_dataset()

    if df is not None:
        included_data, excluded_data = preprocess_data(df)
        preset_key, selected_features, feature_importance_file = select_features_step(df)
        target_feature = select_target_step(df)

        if preset_key or selected_features:
            manually_selected_features = selected_features if selected_features else None
            X_included = select_features(included_data, preset_key, bool(selected_features), manually_selected_features, feature_importance_file)
            y_included = included_data[target_feature]
            st.header("Selected Features")
            display_features_list = X_included.columns.tolist()
            with st.expander(f"View Selected Features ({len(display_features_list)})"):
                for i in display_features_list:
                    st.markdown("- " + i)

            st.sidebar.header("Step 3: Choose Desired Output")
            output = st.sidebar.radio("Output",["Run Machine Learning", "Perform Data Analysis"])

            if output == "Run Machine Learning":
                model_training_step(X_included, y_included, included_data, excluded_data)

            elif output == "Perform Data Analysis":
                data_analysis_step(X_included, y_included, included_data, excluded_data, selected_features)
        else:
            st.warning("No features selected. Please choose a preset, upload importance file, or select manually.")


if __name__ == "__main__":
    main()
