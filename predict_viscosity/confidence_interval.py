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
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Calculate confidence interval
def calculate_confidence_interval(X_included, y_included, NUM_RUNS, model_name="catboost"):
    r2_scores = []

    models = {
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

    if model_name not in models:
        raise ValueError(f"Model '{model_name}' not recognized. Please choose from: {list(models.keys())}")

    for i in tqdm(range(NUM_RUNS), desc="Training and Evaluating"):
        X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i * 4)

        model = models[model_name]
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)
    confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(NUM_RUNS))

    return mean_r2, confidence_interval, r2_scores


def plot_confidence_interval(r2_scores, confidence_interval, mean_r2):
    plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3, label=f"95% CI: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
    plt.axhline(mean_r2, color='green', linestyle='-', label=f"Mean R²: {mean_r2:.2f}")

    plt.title("R² Scores Across Runs with 95% Confidence Interval")
    plt.xlabel("Run Index")
    plt.ylabel("R² Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
