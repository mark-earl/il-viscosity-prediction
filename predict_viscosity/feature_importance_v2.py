import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectFromModel
import seaborn as sns

# Load the data
data_path = 'data/xlsx/working-ils_v2.xlsx'
df = pd.read_excel(data_path)

# Specify columns to drop
cols_to_drop = ['Cation', 'Anion', 'Reference Viscosity', 'Excluded IL', 'T / K']  # Example columns
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# Drop rows with missing target variable
df = df.dropna(subset=['Reference Viscosity Log'])

# Separate features and target
X = df.drop('Reference Viscosity Log', axis=1)
y = df['Reference Viscosity Log']

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Encode categorical columns (if any)
X_encoded = pd.get_dummies(X)

# Train a Random Forest model to compute feature importance
model = CatBoostRegressor()
model.fit(X_encoded, y)

# Extract feature importance
feature_importance = pd.DataFrame({
    'Feature': X_encoded.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance.head(100), x='Importance', y='Feature', palette='viridis')
plt.title('Top 100 Features for Predicting Reference Viscosity Log')
plt.tight_layout()
plt.show()

# Save the feature importance list
feature_importance.to_excel('data/xlsx/feature_importance.xlsx', index=False)
print("Feature importance analysis saved to feature_importance.xlsx.")
