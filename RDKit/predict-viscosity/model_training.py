from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

# Train-test split for included data
def split_data(X_included, y_included):
    return train_test_split(X_included, y_included, test_size=0.2, random_state=0)

# Initialize and train the CatBoost model
def train_model(X_train, y_train):
    model = CatBoostRegressor(
        # iterations=1000,
        # learning_rate=0.1,
        # depth=6,
        verbose=0
    )
    model.fit(X_train, y_train)
    return model

# Predict on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2_rand = r2_score(y_test, y_pred)
    return y_pred, r2_rand
