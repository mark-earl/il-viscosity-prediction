import matplotlib.pyplot as plt
import pandas as pd
import json

def calculate_feature_importance(model, X_included, NUM_FEATURES, output_path="feature_importances.json"):
    feature_importances = model.get_feature_importance()
    feature_names = X_included.columns

    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)

    # Export to JSON
    top_features = feature_importance_df.head(NUM_FEATURES).to_dict(orient='records')
    with open(output_path, 'w') as f:
        json.dump(top_features, f, indent=4)

    print(f"Feature importances exported to {output_path}")
    return feature_importance_df

def plot_feature_importance(feature_importance_df, NUM_FEATURES):
    plt.figure(figsize=(12, 8))
    plt.barh(
        feature_importance_df['Feature'].head(NUM_FEATURES)[::-1],  # Reverse order for horizontal bar plot
        feature_importance_df['Importance'].head(NUM_FEATURES)[::-1],
        color='skyblue'
    )
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature Name')
    plt.title(f'Top {NUM_FEATURES} Important Features')
    plt.tight_layout()
    plt.show()
