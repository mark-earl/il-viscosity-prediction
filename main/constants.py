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
