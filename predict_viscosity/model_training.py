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

# Train-test split for included data
def split_data(X_included, y_included):
    return train_test_split(X_included, y_included, test_size=0.2, random_state=0)

# Initialize and train the selected model
def train_model(X_train, y_train, model_name="catboost"):
    models = {
        "catboost": CatBoostRegressor(verbose=0),
        "xgboost": XGBRegressor(eval_metric='rmse', use_label_encoder=False),
        "random_forest": RandomForestRegressor(),
        "lightgbm": LGBMRegressor(),
        "gradient_boosting": GradientBoostingRegressor(),
        "adaboost": AdaBoostRegressor(),
        "linear_regression": LinearRegression(),
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

    model = models[model_name]
    model.fit(X_train, y_train)
    return model

# Predict on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2_rand = r2_score(y_test, y_pred)
    return y_pred, r2_rand
