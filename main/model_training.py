import optuna
import json
import streamlit as st
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from hyperparameters import MODEL_HYPERPARAMETERS
from sklearn.model_selection import train_test_split
from hyperparameters import MODEL_HYPERPARAMETERS
from visualization import (
    plot_results,
    plot_confidence_interval
)
from confidence_interval_utils import (
    calculate_confidence_interval,
    run_single_committee_model
)
from constants import MODELS

def model_training_step(X_included, y_included, included_data, excluded_data):
    """Main function to execute the model training step."""
    use_committee, model_key, hyperparameters = get_model_selection(X_included, y_included)

    run_ci = st.sidebar.checkbox("Generate Confidence Interval")
    num_runs = st.sidebar.slider("Number of Runs", min_value=0, max_value=250, value=1, step=5) if run_ci else 0

    st.sidebar.header("Step 5: Train Model")
    if st.sidebar.button("Train Model"):
        train_and_evaluate(X_included, y_included, included_data, excluded_data, use_committee, model_key, hyperparameters, run_ci, num_runs)

def get_model_selection(X_included, y_included):
    """Handles model selection and hyperparameter tuning options.

    Returns: use_committee, model_key, hyperparameters
    """
    st.sidebar.header("Step 4: Select Model")
    use_committee = st.sidebar.checkbox("Use Committee")

    # Currently, no hyperparameters for committee
    if use_committee:
        committees = st.sidebar.multiselect("Select Committee Models", MODELS.values(), placeholder="Select Models")
        committee_keys = [key for key, value in MODELS.items() if value in committees]
        return use_committee, committee_keys, {}

    model_name = st.sidebar.selectbox("Select Model", list(MODELS.values()))
    model_key = next(key for key, value in MODELS.items() if value == model_name)

    use_default_hyperparams = st.sidebar.checkbox("Use Default Hyperparameters", value=True)
    hyperparameters = {}
    if not use_default_hyperparams:
        auto_tune = st.sidebar.checkbox("Compute Optimal Hyperparameters")
        if auto_tune:
            n_trials = st.sidebar.slider("Number of Trials", min_value=10, max_value=100, value=30, step=10)
            if st.sidebar.button("Optimize Hyperparameters"):
                st.write("🔍 Optimizing hyperparameters... This may take a while.")

                hyperparameters = optimize_hyperparameters(model_key, X_included, y_included, n_trials)
                st.success("Optimization Complete! Best Parameters Found.")

                # Convert parameters to JSON and allow download
                params_json = json.dumps(hyperparameters, indent=4)
                st.download_button("📥 Download Best Hyperparameters", params_json, "best_hyperparameters.json", "application/json")

                st.json(hyperparameters)
        else:
            hyperparameters = get_manual_hyperparameters(model_key)

    return use_committee, model_key, hyperparameters

def train_and_evaluate(X_included, y_included, included_data, excluded_data, use_committee, models_keys, hyperparameters, run_ci, num_runs):
    """Handles the model training and evaluation logic."""
    X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, random_state=0)

    model=None

    if use_committee:
        if run_ci:
            mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(
                X_included, y_included, num_runs, committee_models=models_keys
            )
            st.pyplot(plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix="(Committee)" if use_committee else ""))
            return

        y_train_pred, y_pred, r2_rand = run_single_committee_model(X_train, X_test, y_train, y_test, models_keys)
        st.write(f"Committee of: {', '.join([MODELS[k] for k in models_keys])} trained successfully!")
    else:
        if run_ci:
            mean_r2, confidence_interval, r2_scores = calculate_confidence_interval(
                X_included, y_included, num_runs, model_name=models_keys
            )
            st.pyplot(plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix="(Committee)" if use_committee else ""))
            return

        model = train_model(X_train, y_train, models_keys, hyperparameters)
        y_train_pred, y_pred = model.predict(X_train), model.predict(X_test)
        r2_rand = model.score(X_test, y_test)

    plot_and_display_results(included_data, excluded_data, X_train, y_train, y_train_pred, y_test, y_pred, r2_rand, y_included, model, models_keys)

def train_model(X_train, y_train, model_name="catboost", hyperparameters={}):
    model_classes = {
        "catboost": CatBoostRegressor,
        "xgboost": XGBRegressor,
        "random_forest": RandomForestRegressor,
        "lightgbm": LGBMRegressor,
        "gradient_boosting": GradientBoostingRegressor,
        "adaboost": AdaBoostRegressor,
        "linear_regression": LinearRegression,
        "ridge": Ridge,
        "lasso": Lasso,
        "elastic_net": ElasticNet,
        "svr": SVR,
        "knn": KNeighborsRegressor,
        "decision_tree": DecisionTreeRegressor,
        "mlp": MLPRegressor
    }

    if model_name not in model_classes:
        st.error(f"Model '{model_name}' not recognized. Please choose from: {list(model_classes.keys())}")

    # Instantiate the model with hyperparameters
    model = model_classes[model_name](**hyperparameters)

    model.fit(X_train, y_train)

    return model

def get_manual_hyperparameters(model_key):
    """Retrieves manual hyperparameter settings from the user."""
    hyperparameters = {}
    if model_key in MODEL_HYPERPARAMETERS:
        st.sidebar.subheader(f"Hyperparameters for {MODELS[model_key]}")
        for param, settings in MODEL_HYPERPARAMETERS[model_key].items():
            param_type, *values = settings
            if param_type == "slider":
                min_val, max_val, default_val, step = values
                hyperparameters[param] = st.sidebar.slider(param, min_val, max_val, default_val, step)
            elif param_type == "selectbox":
                options, default_val = values
                hyperparameters[param] = st.sidebar.selectbox(param, options, index=options.index(default_val))
    return hyperparameters

def plot_and_display_results(included_data, excluded_data, X_train, y_train, y_train_pred, y_test, y_pred, r2_rand, y_included, model=None, model_keys=None):
    """Handles plotting and displaying the results."""
    st.header("R² Score on Test Data")
    st.markdown(f"<p style='font-size:40px;color:#0096FF;'>{r2_rand:.3f}</p>", unsafe_allow_html=True)

    y_excluded, y_excluded_pred = None, None
    if excluded_data is not None and not excluded_data.empty:
        X_excluded = excluded_data.drop(columns=[y_included.name])
        X_excluded = X_excluded[X_train.columns]
        y_excluded = excluded_data[y_included.name]

    if model:
        y_excluded_pred = model.predict(X_excluded)

    elif model_keys:
        _, y_excluded_pred, _ = run_single_committee_model(X_train, X_excluded, y_train, y_excluded, model_keys)

    st.pyplot(plot_results(y_train, y_train_pred, y_test, y_pred, r2_rand, y_excluded, y_excluded_pred))

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2_rand = r2_score(y_test, y_pred)
    return y_pred, r2_rand

def objective(trial, model_key, X_train, y_train):
    """Objective function for hyperparameter tuning using Optuna."""
    param_space = MODEL_HYPERPARAMETERS.get(model_key, {})

    hyperparameters = {}
    for param, settings in param_space.items():
        param_type, *values = settings
        if param_type == "slider":
            min_val, max_val, default_val, step = values
            if isinstance(step, int):  # Ensure integers are correctly suggested
                hyperparameters[param] = trial.suggest_int(param, int(min_val), int(max_val), int(step))
            else:
                hyperparameters[param] = trial.suggest_float(param, min_val, max_val, step=step)
        elif param_type == "selectbox":
            options, default_val = values
            hyperparameters[param] = trial.suggest_categorical(param, options)

    model = train_model(X_train, y_train, model_key, hyperparameters)
    return model.score(X_train, y_train)  # Maximizing R² score

def optimize_hyperparameters(model_key, X_train, y_train, n_trials=30):
    """Run hyperparameter optimization and return best parameters with a progress bar."""
    study = optuna.create_study(direction="maximize")

    progress_bar = st.progress(0)  # Initialize progress bar
    progress_text = st.empty()  # Placeholder for updating text

    def callback(study, trial):
        """Update progress bar and text dynamically."""
        progress = (trial.number + 1) / n_trials  # Normalize progress (0 to 1)
        progress_bar.progress(progress)
        progress_text.text(f"Optimization Progress: {trial.number + 1}/{n_trials} trials completed...")

    study.optimize(lambda trial: objective(trial, model_key, X_train, y_train), n_trials=n_trials, callbacks=[callback])

    progress_bar.empty()  # Remove progress bar when finished

    return study.best_params
