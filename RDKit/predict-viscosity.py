import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Step 1: Load the data from the Excel file
data_path = 'RDKit/data/heavyweight-ils.xlsx'
df = pd.read_excel(data_path)

# Drop unnecessary columns
df = df.drop(columns=[
    'IL ID', 'Cation', 'Anion',
    'cation_Ion type', 'cation_Family', 'cation_Number of ILs composed of the ion',
    'anion_Ion type', 'anion_Family', 'anion_Number of ILs composed of the ion'
])

# Step 4: Define features and target variable
# X = df.drop(columns=['η / mPa s'])  # Features
X = df.drop(columns=['η / mPa s'])  # Features
y = df['η / mPa s']  # Target

# Step 5: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Define feature weights for multiple attributes
# Features to prioritize: 'T / K' and 'P / bar'
feature_weights_map = {
    'T / K': 2,
    'anion_Molecular Surface Area': 2,
    'anion_Molecular Volume': 2,
    'anion_Molecular Radius': 2,
    'cation_Molecular Surface Area': 2,
    'cation_Molecular Volume': 2,
    'cation_Molecular Radius': 2,
}

# Assign weights for all features based on the mapping
feature_weights = [feature_weights_map.get(feature, 1) for feature in X_train.columns]

# Step 7: Initialize and train the CatBoost model with feature weights
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=6,
    verbose=200,
    # feature_weights=feature_weights  # Apply custom weights
)
model.fit(X_train, y_train)

# Step 8: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 9: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error: {mse}")
print(f"R²: {r2}")

# Plot predicted vs. actual values (Before Filtering)
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.scatter(y_pred, y_test, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)  # Ideal line
plt.title("Predicted vs Actual (Before Filtering)")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")

plt.tight_layout()
plt.show()
