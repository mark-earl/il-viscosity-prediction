from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Models Dictionary
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor

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

    if committee_models:
        # Use committee mode
        models = [MODEL_REGISTRY[model] for model in committee_models]
    else:
        # Use single model
        if model_name not in MODEL_REGISTRY:
            raise ValueError(f"Model '{model_name}' not recognized. Available models: {list(MODEL_REGISTRY.keys())}")
        models = [MODEL_REGISTRY[model_name]]

    for i in tqdm(range(num_runs), desc="Training and Evaluating Models"):
        X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i * 4)

        # Store predictions for ensemble averaging
        committee_predictions = np.zeros_like(y_test, dtype=float) if committee_models else None

        for model in models:
            y_pred, r2 = train_and_evaluate_model(X_train, X_test, y_train, y_test, model)
            if committee_models:
                committee_predictions += y_pred / len(models)

        r2_final = r2_score(y_test, committee_predictions) if committee_models else r2
        r2_scores.append(r2_final)

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)
    confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(num_runs))

    return mean_r2, confidence_interval, r2_scores


def plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix=""):
    """Plots R² scores with a confidence interval."""
    fig = plt.figure()
    plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3,
                 label=f"95% CI: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
    plt.axhline(mean_r2, color='green', linestyle='-', label=f"Mean R²: {mean_r2:.2f}")

    plt.title(f"R² Scores Across Runs with 95% Confidence Interval {title_suffix}")
    plt.xlabel("Run Index")
    plt.ylabel("R² Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig

def run_single_committee_model(X_train, X_test, y_train, y_test, models):
    # Filter and retrieve the models from MODEL_REGISTRY based on the keys in `models`
    selected_models = {name: MODEL_REGISTRY[name] for name in models if name in MODEL_REGISTRY}

    if not selected_models:
        raise ValueError("No valid models found for the provided keys. Please check the model names.")

    committee_predictions = np.zeros_like(y_test, dtype=float)

    # Train and predict with each selected model in the committee
    for name, model in selected_models.items():
        model.fit(X_train, y_train)
        committee_predictions += model.predict(X_test) / len(selected_models)

    r2_rand = r2_score(y_test, committee_predictions)
    return committee_predictions, r2_rand
