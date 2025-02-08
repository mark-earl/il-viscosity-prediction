import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import streamlit as st
from catboost import CatBoostRegressor

def plot_feature_importance(X, y, num_features):
    """Plots and allows download of top feature importance using CatBoostRegressor."""
    progress = st.progress(0)

    model = CatBoostRegressor(verbose=0)
    model.fit(X, y)
    progress.progress(50)

    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    # Plot the feature importance
    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(num_features), x='Importance', y='Feature', palette='viridis')
    plt.title(f'Top {num_features} Features for Target Prediction')
    plt.tight_layout()
    st.pyplot(plt.gcf())
    plt.clf()

    # Create a DataFrame only for the top selected features
    top_features_df = feature_importance.head(num_features)

    # Allow the user to download the feature importance as an Excel file
    buffer = io.BytesIO()
    top_features_df.to_excel(buffer, index=False, engine='xlsxwriter')
    buffer.seek(0)

    st.download_button(
        label="Download Top Features as Excel",
        data=buffer,
        file_name=f"top_{num_features}_features.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    progress.progress(100)


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

    # Select the top num_features for the heatmap
    selected_features = feature_importance['Feature'].head(num_features).tolist()
    heatmap_df = X[selected_features].copy()

    plt.figure(figsize=(12, 8))
    correlation_matrix = heatmap_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title(f"Correlation Heatmap of Top {num_features} Features")
    st.pyplot(plt.gcf())
    plt.clf()

    progress.progress(100)


def plot_graph_relationships(X):
    """Generates a graph where rows are nodes and relationships between them are edges."""
    progress = st.progress(0)
    G = nx.Graph()

    # Add nodes (each row is a node)
    for i, row_data in X.iterrows():
        G.add_node(i, features=row_data.to_dict())

    total_edges = 0
    for i in range(len(X)):
        for j in range(i + 1, len(X)):
            distance = np.linalg.norm(X.iloc[i].values - X.iloc[j].values)
            if distance < np.percentile(distance, 5):  # Add an edge if similarity is high
                G.add_edge(i, j, weight=distance)
                total_edges += 1
        progress.progress(int((i + 1) / len(X) * 50))

    pos = nx.spring_layout(G)
    plt.figure(figsize=(10, 8))
    nx.draw(G, pos, with_labels=False, node_size=50, node_color="skyblue", edge_color="gray")
    plt.title("Graph Relationships between Rows")
    st.pyplot(plt.gcf())
    plt.clf()
    progress.progress(100)
