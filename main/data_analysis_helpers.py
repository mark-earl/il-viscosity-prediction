import streamlit as st
import pandas as pd
from constants import MODELS_WITH_FEATURE_IMPORTANCE

def _get_model_by_key(model_key):
    """
    Helper function to return the model based on the key.
    """
    if model_key == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(verbose=0)
    elif model_key == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor()
    elif model_key == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor()
    elif model_key == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor()
    elif model_key == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor()
    elif model_key == "adaboost":
        from sklearn.ensemble import AdaBoostRegressor
        return AdaBoostRegressor()
    elif model_key == "decision_tree":
        from sklearn.tree import DecisionTreeRegressor
        return DecisionTreeRegressor()
    else:
        raise ValueError(f"Unsupported model key: {model_key}")

def compute_feature_importance(X, y, use_committee, committee_models, model_key):
    """Compute feature importance based on a single model or a committee of models."""
    if use_committee and committee_models:
        feature_importance_sum = pd.Series(0, index=X.columns, dtype=float)

        for model_key in committee_models:
            model = _get_model_by_key(model_key)
            model.fit(X, y)
            feature_importance_sum += pd.Series(model.feature_importances_, index=X.columns)

        return feature_importance_sum / len(committee_models)
    else:
        if not model_key:
            st.error("Please select a model for feature analysis.")
            return None
        model = _get_model_by_key(model_key)
        model.fit(X, y)
        return pd.Series(model.feature_importances_, index=X.columns)

def get_model_selection():
    """Handles model selection from Streamlit UI."""
    use_committee = st.sidebar.checkbox("Use Committee")

    if use_committee:
        committees = st.sidebar.multiselect("Select Committee Models", MODELS_WITH_FEATURE_IMPORTANCE.values(), placeholder="Select Models")
        committee_keys = [key for key, value in MODELS_WITH_FEATURE_IMPORTANCE.items() if value in committees]
        return use_committee, committee_keys, None
    else:
        model_name = st.sidebar.selectbox("Select Model", list(MODELS_WITH_FEATURE_IMPORTANCE.values()))
        model_key = next(key for key, value in MODELS_WITH_FEATURE_IMPORTANCE.items() if value == model_name)
        return use_committee, None, model_key
