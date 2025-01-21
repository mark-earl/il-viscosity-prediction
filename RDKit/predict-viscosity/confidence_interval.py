from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
import numpy as np
from sklearn.metrics import r2_score
from tqdm import tqdm
from scipy.stats import norm
import matplotlib.pyplot as plt

# Calculate confidence interval
def calculate_confidence_interval(X_included, y_included, NUM_RUNS):
    r2_scores = []

    for i in tqdm(range(NUM_RUNS), desc="Training and Evaluating"):
        X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i)

        model = CatBoostRegressor(
            # iterations=1000,
            # learning_rate=0.1,
            # depth=6,
            verbose=0
        )
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
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3, label="95% CI")
    plt.axhline(mean_r2, color='green', linestyle='-', label="Mean R²")
    plt.title(f"R² Scores Across Runs with 95% Confidence Interval")
    plt.xlabel("Run Index")
    plt.ylabel("R² Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
