import streamlit as st
from upload_dataset import *
from constants import *
from data_preprocessing import *
from feature_selection import *
from model_training import train_model, split_data, optimize_hyperparameters, objective
from hyperparameters import MODEL_HYPERPARAMETERS
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

def show_features(X_included):
    st.header("Selected Features")
    features_list = X_included.columns.tolist()
    with st.expander(f"View Selected Features ({len(features_list)})"):
        for i in features_list:
            st.markdown("- " + i)

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

def model_training_step(X_included, y_included, included_data, excluded_data):
    st.sidebar.header("Step 4: Select Model")
    use_committee = st.sidebar.checkbox("Use Committee")

    hyperparameters = {}

    if use_committee:
        committees = st.sidebar.multiselect("Select Committee Models", MODELS.values(), placeholder="Select Models")
        committee_keys = [key for key, value in MODELS.items() if value in committees]
    else:
        model_name = st.sidebar.selectbox("Select Model", list(MODELS.values()))
        model_key = next(key for key, value in MODELS.items() if value == model_name)

        use_default_hyperparams = st.sidebar.checkbox("Use Default Hyperparameters", value=True)

        if not use_default_hyperparams:
            auto_tune = st.sidebar.checkbox("Automatically Optimize Hyperparameters")

            if auto_tune:
                n_trials = st.sidebar.slider("Number of Trials", min_value=10, max_value=100, value=30, step=10)
            else:
                if model_key in MODEL_HYPERPARAMETERS:
                    st.sidebar.subheader(f"Hyperparameters for {model_name}")

                    for param, settings in MODEL_HYPERPARAMETERS[model_key].items():
                        param_type, *values = settings

                        if param_type == "slider":
                            min_val, max_val, default_val, step = values
                            hyperparameters[param] = st.sidebar.slider(param, min_val, max_val, default_val, step)

                        elif param_type == "selectbox":
                            options, default_val = values
                            hyperparameters[param] = st.sidebar.selectbox(param, options, index=options.index(default_val))

    # Confidence Interval Option
    run_ci = st.sidebar.checkbox("Generate Confidence Interval")
    num_runs = st.sidebar.slider("Number of Runs", min_value=0, max_value=250, value=1, step=5) if run_ci else 0

    st.sidebar.header("Step 5: Train Model")

    if st.sidebar.button("Train Model"):
        X_train, X_test, y_train, y_test = split_data(X_included, y_included)

        if run_ci:
            # Run confidence interval calculation
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
            # Train the model without confidence interval
            if use_committee:
                y_train_pred, y_pred, r2_rand = run_single_committee_model(X_train, X_test, y_train, y_test, committee_keys)
                st.write(f"Committee of: {', '.join(committees)} trained successfully!")
                st.pyplot(plot_results(included_data, excluded_data, y_test, y_pred, r2_rand))
            else:
                model = train_model(X_train, y_train, model_key, hyperparameters if not use_default_hyperparams else {})
                y_train_pred, y_pred = model.predict(X_train), model.predict(X_test)
                r2_rand = model.score(X_test, y_test)

                # Predict excluded data if available
                if excluded_data is not None and not excluded_data.empty:
                    X_excluded = excluded_data.drop(columns=[y_included.name], errors='ignore')
                    y_excluded = excluded_data[y_included.name]
                    y_excluded_pred = model.predict(X_excluded)
                else:
                    y_excluded, y_excluded_pred = None, None

                st.header("R² Score on Test Data")
                st.markdown(f"<p style='font-size:40px;color:#0096FF;'>{r2_rand:.3f}</p>", unsafe_allow_html=True)
                st.pyplot(plot_results(included_data, excluded_data, y_test, y_pred, r2_rand, y_excluded, y_excluded_pred))

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

    if not select_features:
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
