import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import streamlit as st
from catboost import CatBoostRegressor

def plot_feature_importance(X, y, num_features, use_committee=False, committee_models=None, model_name=None):
    """
    Plots and allows download of top feature importance using a selected model or a committee of models.
    """

    progress = st.progress(0)

    # Initialize the appropriate model(s)
    model = None
    if use_committee and committee_models:
        feature_importance_sum = pd.Series(0, index=X.columns, dtype=float)

        for model_key in committee_models:
            model = _get_model_by_key(model_key)
            model.fit(X, y)
            feature_importance_sum += pd.Series(model.feature_importances_, index=X.columns)

        feature_importance_avg = feature_importance_sum / len(committee_models)
    else:
        if not model_name:
            st.error("Please select a model for feature analysis.")
            return
        model = _get_model_by_key(model_name)
        model.fit(X, y)
        feature_importance_avg = pd.Series(model.feature_importances_, index=X.columns)

    # Prepare DataFrame for plotting
    feature_importance = feature_importance_avg.sort_values(ascending=False).head(num_features).reset_index()
    feature_importance.columns = ['Feature', 'Importance']

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {num_features} Features for Target Prediction')
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Save plot as PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download Feature Importance Plot as PNG",
        data=buffer,
        file_name=f"top_{num_features}_features_plot.png",
        mime="image/png"
    )
    progress.progress(100)


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

def plot_correlation_heatmap(X, y, num_features):
    """Generates a correlation heatmap of the selected number of important features."""
    progress = st.progress(0)

    # Calculate feature importance using CatBoostRegressor
    model = CatBoostRegressor(verbose=0)
    model.fit(X, y)
    progress.progress(30)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    selected_features = feature_importance['Feature'].head(num_features).tolist()
    heatmap_df = X[selected_features].copy()

    plt.figure(figsize=(12, 8))
    correlation_matrix = heatmap_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title(f"Correlation Heatmap of Top {num_features} Features")
    st.pyplot(plt.gcf())

    # Save plot as PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download Heatmap as PNG",
        data=buffer,
        file_name=f"correlation_heatmap_{num_features}_features.png",
        mime="image/png"
    )
    progress.progress(100)


def plot_graph_relationships(X, selected_features):
    """
    Generates a graph where rows are nodes and relationships between them are edges based on user-selected features.

    Args:
        X (DataFrame): The input data.
        selected_features (list): List of user-selected features to use for computing relationships.
    """
    progress = st.progress(0)
    G = nx.Graph()

    if not selected_features:
        st.error("Please select at least one feature.")
        return

    # Filter the DataFrame based on the selected features
    feature_data = X[selected_features]

    # Add nodes (each row is a node)
    for i, row_data in feature_data.iterrows():
        G.add_node(i, features=row_data.to_dict(), label=str(i))

    # Create edges based on feature similarity (Euclidean distance)
    total_edges = 0
    for i in range(len(feature_data)):
        for j in range(i + 1, len(feature_data)):
            distance = np.linalg.norm(feature_data.iloc[i].values - feature_data.iloc[j].values)
            if distance < np.percentile(distance, 25):  # Add an edge if distance is in the closest 25% percentile
                G.add_edge(i, j, weight=distance, label=f"{distance:.2f}")
                total_edges += 1
        progress.progress(int((i + 1) / len(feature_data) * 50))

    # Create a spring layout for graph visualization
    pos = nx.spring_layout(G, seed=42)

    # Draw the graph
    plt.figure(figsize=(12, 8))

    # Draw nodes with labels
    nx.draw_networkx_nodes(G, pos, node_size=300, node_color="skyblue")
    nx.draw_networkx_labels(G, pos, labels=nx.get_node_attributes(G, 'label'), font_size=12, font_weight='bold')

    # Draw edges with labels
    nx.draw_networkx_edges(G, pos, edge_color="gray")
    edge_labels = nx.get_edge_attributes(G, 'label')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=10)

    plt.title("Graph Relationships between Rows based on Selected Features")
    st.pyplot(plt.gcf())

    # Save plot as PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download Graph Plot as PNG",
        data=buffer,
        file_name="graph_relationships_selected_features_plot.png",
        mime="image/png"
    )
    plt.clf()
    progress.progress(100)
