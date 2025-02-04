from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score
from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Define the committee of models
MODELS = {
    "catboost": CatBoostRegressor(verbose=0),
    "linear_regression": LinearRegression(),
    "ridge": Ridge(),
    "xgboost": XGBRegressor(eval_metric='rmse', use_label_encoder=False),
}


def calculate_committee_confidence_interval(X_included, y_included, NUM_RUNS):
    r2_scores = []

    for i in tqdm(range(NUM_RUNS), desc="Training and Evaluating Committee Models"):
        X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i * 4)

        # Store predictions for ensemble averaging
        committee_predictions = np.zeros_like(y_test, dtype=float)

        # Train and predict with each model in the committee
        for name, model in MODELS.items():
            model.fit(X_train, y_train)
            committee_predictions += model.predict(X_test) / len(MODELS)

        # Compute R^2 score for the ensemble
        r2 = r2_score(y_test, committee_predictions)
        r2_scores.append(r2)

    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores, ddof=1)
    confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(NUM_RUNS))

    return mean_r2, confidence_interval, r2_scores


def plot_committee_confidence_interval(r2_scores, confidence_interval, mean_r2):
    plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3, label=f"95% CI: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
    plt.axhline(mean_r2, color='green', linestyle='-', label=f"Mean R²: {mean_r2:.2f}")

    plt.title("R² Scores Across Runs with 95% Confidence Interval (Committee)")
    plt.xlabel("Run Index")
    plt.ylabel("R² Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
