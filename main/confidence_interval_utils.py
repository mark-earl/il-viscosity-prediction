import numpy as np
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import norm

# Define available models
MODEL_REGISTRY = {
    "catboost": CatBoostRegressor(verbose=0),
    "xgboost": XGBRegressor(),
    "random_forest": RandomForestRegressor(),
    "lightgbm": LGBMRegressor(),
    "gradient_boosting": GradientBoostingRegressor(),
    "adaboost": AdaBoostRegressor(),
    "linear_regression": LinearRegression(n_jobs=-1),
    "ridge": Ridge(),
    "lasso": Lasso(),
    "elastic_net": ElasticNet(),
    "svr": SVR(),
    "knn": KNeighborsRegressor(),
    "decision_tree": DecisionTreeRegressor(),
    "mlp": MLPRegressor(max_iter=500)
}

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model):
    """Trains and evaluates a single model."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    return y_pred, r2

def calculate_confidence_interval(X_included, y_included, num_runs, model_name=None, committee_models=None):
    """Calculates confidence intervals for a single model or committee ensemble."""
    r2_scores = []
    progress_bar = st.progress(0)

    if committee_models:
        models = [MODEL_REGISTRY[model] for model in committee_models]
    else:
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(MODEL_REGISTRY.keys())}")
        models = [MODEL_REGISTRY[model_name]]

    for i in range(num_runs):
        X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i * 4)

        # Store predictions for ensemble averaging
        committee_predictions = np.zeros_like(y_test, dtype=float) if committee_models else None

        for model in models:
            y_pred, r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test, model)
            if committee_models:
                committee_predictions += y_pred / len(models)

        r2_final = r2_score(y_test, committee_predictions) if committee_models else r2
        r2_scores.append(r2_final)

        progress_bar.progress(int((i + 1) / num_runs * 100))

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)
    confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(num_runs))

    progress_bar.empty()
    return mean_r2, confidence_interval, r2_scores

def run_single_committee_model(X_train, X_test, y_train, y_test, models):
    """Trains and evaluates a committee model."""
    selected_models = {name: MODEL_REGISTRY[name] for name in models if name in MODEL_REGISTRY}

    if not selected_models:
        raise ValueError("No valid models found for the provided keys. Please check the model names.")

    committee_predictions = np.zeros_like(y_test, dtype=float)
    progress_bar = st.progress(0)

    for i, (name, model) in enumerate(selected_models.items()):
        model.fit(X_train, y_train)
        committee_predictions += model.predict(X_test) / len(selected_models)
        progress_bar.progress(int((i + 1) / len(selected_models) * 100))

    progress_bar.empty()
    r2_rand = r2_score(y_test, committee_predictions)
    return committee_predictions, r2_rand
