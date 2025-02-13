MODEL_HYPERPARAMETERS = {
    "catboost": {
        "iterations": ("slider", 100, 5000, 1000, 100),
        "learning_rate": ("slider", 0.001, 0.3, 0.1, 0.01),
        "depth": ("slider", 2, 16, 6, 1)
    },
    "xgboost": {
        "n_estimators": ("slider", 50, 1000, 100, 50),
        "learning_rate": ("slider", 0.01, 0.3, 0.1, 0.01),
        "max_depth": ("slider", 2, 16, 6, 1),
        "subsample": ("slider", 0.5, 1.0, 0.8, 0.05)
    },
    "random_forest": {
        "n_estimators": ("slider", 50, 500, 100, 50),
        "max_depth": ("slider", 2, 20, 10, 1),
        "min_samples_split": ("slider", 2, 10, 2, 1),
        "min_samples_leaf": ("slider", 1, 10, 1, 1)
    },
    "lightgbm": {
        "num_leaves": ("slider", 10, 100, 31, 5),
        "learning_rate": ("slider", 0.01, 0.3, 0.1, 0.01),
        "n_estimators": ("slider", 50, 1000, 100, 50)
    },
    "linear_regression": {},  # No hyperparameters to tune
    "ridge": {
        "alpha": ("slider", 0.01, 10.0, 1.0, 0.01)
    },
    "lasso": {
        "alpha": ("slider", 0.01, 10.0, 1.0, 0.01)
    },
    "svr": {
        "C": ("slider", 0.1, 10.0, 1.0, 0.1),
        "epsilon": ("slider", 0.01, 1.0, 0.1, 0.01)
    },
    "knn": {
        "n_neighbors": ("slider", 1, 20, 5, 1),
        "weights": ("selectbox", ["uniform", "distance"], "uniform")
    }
}
