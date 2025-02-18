# Features to exclude for Excel file "working-ils.xlsx"
IRRELEVANT_FEATURES_WORKING_ILS = [
    'IL ID',
    'Cation',
    'Anion',
    'Excluded IL',
    'Reference Viscosity'
]

FEATURE_PRESET_OPTIONS = {
    "functional_groups": "Functional Groups",
    "molecular_descriptors": "Molecular Descriptors",
    "both": "Both"
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

MODEL_HYPERPARAMETERS = {
    "catboost": {
        "iterations": ("slider", 100, 5000, 1000, 100),
        "learning_rate": ("slider", 0.001, 0.3, 0.1, 0.01),
        "depth": ("slider", 2, 16, 6, 1),
        "l2_leaf_reg": ("slider", 1, 10, 3, 1),
        "bagging_temperature": ("slider", 0.0, 1.0, 0.8, 0.1)
    },
    "xgboost": {
        "n_estimators": ("slider", 50, 1000, 100, 50),
        "learning_rate": ("slider", 0.01, 0.3, 0.1, 0.01),
        "max_depth": ("slider", 2, 16, 6, 1),
        "subsample": ("slider", 0.5, 1.0, 0.8, 0.05),
        "colsample_bytree": ("slider", 0.5, 1.0, 0.8, 0.05),
        "gamma": ("slider", 0, 10, 0, 1)
    },
    "random_forest": {
        "n_estimators": ("slider", 50, 500, 100, 50),
        "max_depth": ("slider", 2, 20, 10, 1),
        "min_samples_split": ("slider", 2, 10, 2, 1),
        "min_samples_leaf": ("slider", 1, 10, 1, 1),
        "max_features": ("selectbox", ["sqrt", "log2"], "sqrt")
    },
    "lightgbm": {
        "num_leaves": ("slider", 10, 100, 31, 5),
        "learning_rate": ("slider", 0.01, 0.3, 0.1, 0.01),
        "n_estimators": ("slider", 50, 1000, 100, 50),
        "subsample": ("slider", 0.5, 1.0, 0.8, 0.05),
        "colsample_bytree": ("slider", 0.5, 1.0, 0.8, 0.05)
    },
    "gradient_boosting": {
        "n_estimators": ("slider", 50, 1000, 100, 50),
        "learning_rate": ("slider", 0.01, 0.3, 0.1, 0.01),
        "max_depth": ("slider", 2, 16, 6, 1),
        "min_samples_split": ("slider", 2, 10, 2, 1),
        "min_samples_leaf": ("slider", 1, 10, 1, 1)
    },
    "adaboost": {
        "n_estimators": ("slider", 50, 1000, 100, 50),
        "learning_rate": ("slider", 0.01, 1.0, 0.1, 0.01)
    },
    "linear_regression": {},  # No hyperparameters to tune
    "ridge": {
        "alpha": ("slider", 0.01, 10.0, 1.0, 0.01)
    },
    "lasso": {
        "alpha": ("slider", 0.01, 10.0, 1.0, 0.01)
    },
    "elastic_net": {
        "alpha": ("slider", 0.01, 10.0, 1.0, 0.01),
        "l1_ratio": ("slider", 0.0, 1.0, 0.5, 0.05)
    },
    "svr": {
        "C": ("slider", 0.1, 10.0, 1.0, 0.1),
        "epsilon": ("slider", 0.01, 1.0, 0.1, 0.01),
        "kernel": ("selectbox", ["linear", "poly", "rbf", "sigmoid"], "rbf")
    },
    "knn": {
        "n_neighbors": ("slider", 1, 20, 5, 1),
        "weights": ("selectbox", ["uniform", "distance"], "uniform"),
        "p": ("slider", 1, 5, 2, 1)
    },
    "decision_tree": {
        "max_depth": ("slider", 2, 20, 10, 1),
        "min_samples_split": ("slider", 2, 10, 2, 1),
        "min_samples_leaf": ("slider", 1, 10, 1, 1)
    },
    "mlp": {
        "hidden_layer_sizes": ("selectbox", [(50,), (100,), (50,50), (100,100)], (50,)),
        "activation": ("selectbox", ["identity", "logistic", "tanh", "relu"], "relu"),
        "solver": ("selectbox", ["lbfgs", "sgd", "adam"], "adam"),
        "alpha": ("slider", 0.0001, 0.1, 0.0001, 0.0001),
        "learning_rate": ("selectbox", ["constant", "invscaling", "adaptive"], "constant")
    }
}
