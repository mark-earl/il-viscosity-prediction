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

    plt.figure(figsize=(10, 6))
    sns.barplot(data=feature_importance.head(num_features), x='Importance', y='Feature', palette='viridis')
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


def plot_graph_relationships(X):
    """Generates a graph where rows are nodes and relationships between them are edges."""
    progress = st.progress(0)
    G = nx.Graph()

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

    # Save plot as PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download Graph Plot as PNG",
        data=buffer,
        file_name="graph_relationships_plot.png",
        mime="image/png"
    )
    plt.clf()
    progress.progress(100)
