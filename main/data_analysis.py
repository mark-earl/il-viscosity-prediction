import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import streamlit as st
from data_analysis_helpers import get_model_selection, compute_feature_importance

def data_analysis_step(X_included, y_included, included_data, excluded_data, selected_features):
    output = st.sidebar.radio("Data Analysis Options", ["Analyze Feature Importance", "Generate Correlational Heatmap", "Generate Graph"])

    if output == "Analyze Feature Importance":
        analyze_features(X_included, y_included)

    if output == "Generate Correlational Heatmap":
        generate_correlational_heatmap(X_included, y_included)

    if output == "Generate Graph":
        generate_graph(X_included, selected_features)

def analyze_features(X_included, y_included):
    num_features = st.sidebar.slider("Number of Features", min_value=1, max_value=len(X_included.columns), value=1)

    use_committee, committee_keys, model_key = get_model_selection()

    if st.sidebar.button("Analyze Features"):
        plot_feature_importance(X_included, y_included, num_features, use_committee, committee_keys, model_key)

def generate_correlational_heatmap(X_included, y_included):
    num_heatmap_features = st.sidebar.slider("Number of Features", min_value=2, max_value=len(X_included.columns), value=10)

    use_committee, committee_keys, model_key = get_model_selection()

    if st.sidebar.button("Generate Heatmap"):
        plot_correlation_heatmap(X_included, y_included, num_heatmap_features, use_committee, committee_keys, model_key)

def generate_graph(X_included, selected_features):
    if st.sidebar.button("Generate Graph"):
        plot_graph_relationships(X_included, selected_features)

def plot_feature_importance(X, y, num_features, use_committee=False, committee_models=None, model_key=None):
    progress = st.progress(0)

    feature_importance_avg = compute_feature_importance(X, y, use_committee, committee_models, model_key)
    if feature_importance_avg is None:
        return  # Stop execution if model selection failed

    # Select the top features
    top_features = feature_importance_avg.nlargest(num_features)

    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(x=top_features.values, y=top_features.index, palette='viridis')
    plt.title(f'Top {num_features} Features for Target Prediction')
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # Save as PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    st.download_button("Download Feature Importance Plot as PNG", buffer, f"top_{num_features}_features.png", "image/png")

    # Save feature data as Excel
    buffer = io.BytesIO()
    top_features.to_frame(name="Importance").reset_index().to_excel(buffer, index=False, engine='xlsxwriter')
    buffer.seek(0)
    st.download_button("Download Top Features as Excel", buffer, f"top_{num_features}_features.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    progress.progress(100)

def plot_correlation_heatmap(X, y, num_features, use_committee=False, committee_models=None, model_key=None):
    progress = st.progress(0)

    feature_importance_avg = compute_feature_importance(X, y, use_committee, committee_models, model_key)
    if feature_importance_avg is None:
        return

    selected_features = feature_importance_avg.nlargest(num_features).index
    heatmap_df = X[selected_features].copy()

    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
    plt.title(f"Correlation Heatmap of Top {num_features} Features")
    st.pyplot(plt.gcf())

    # Save plot as PNG
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    st.download_button("Download Heatmap as PNG", buffer, f"correlation_heatmap_{num_features}_features.png", "image/png")

    progress.progress(100)

def plot_graph_relationships(X, selected_features):

    progress = st.progress(0)
    G = nx.Graph()

    # Filter the DataFrame based on the selected features
    feature_data = X[selected_features]

    # Randomly sample 50 nodes such that computation is doable
    subset = feature_data.sample(50)

    # Add nodes (each row is a node)
    for i, row_data in subset.iterrows():
        G.add_node(i, features=row_data.to_dict(), label=str(i))

    # Create edges based on feature similarity (Euclidean distance)
    total_edges = 0
    for i in range(len(subset)):
        for j in range(i + 1, len(subset)):
            distance = np.linalg.norm(subset.iloc[i].values - subset.iloc[j].values)
            if distance < np.percentile(distance, 25):  # Add an edge if distance is in the closest 25% percentile
                G.add_edge(i, j, weight=distance, label=f"{distance:.2f}")
                total_edges += 1
        progress.progress(int((i + 1) / len(subset) * 50))

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
