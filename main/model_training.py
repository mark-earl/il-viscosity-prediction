import optuna
import streamlit as st
from sklearn.model_selection import train_test_split
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

# Train-test split for included data
def split_data(X_included, y_included):
    return train_test_split(X_included, y_included, test_size=0.2, random_state=0)

# Initialize and train the selected model with hyperparameters
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
        raise ValueError(f"Model '{model_name}' not recognized. Please choose from: {list(model_classes.keys())}")

    # Instantiate the model with hyperparameters
    model = model_classes[model_name](**hyperparameters)
    model.fit(X_train, y_train)

    return model


# Predict on the test set
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
