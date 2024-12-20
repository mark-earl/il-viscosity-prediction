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

# Step 2: Identify and filter out outliers in the target variable ('η / mPa s')
def filter_outliers(data, column, lower_multiplier=1.5, upper_multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - lower_multiplier * IQR
    upper_bound = Q3 + upper_multiplier * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

df = filter_outliers(df, 'η / mPa s')

# Step 3: Define features and target variable
X = df.drop(columns=['η / mPa s'])  # Features
y = df['η / mPa s']  # Target

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Initialize and train the CatBoost model
model = CatBoostRegressor(
    iterations=10000,
    learning_rate=0.01,
    depth=10,
    verbose=200
)
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R²: {r2}")

# Step 8: Filter outliers in residuals
residuals = y_test - y_pred

def filter_outlier_predictions(y_test, y_pred, lower_multiplier=1.5, upper_multiplier=1.5):
    residuals = y_test - y_pred
    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - lower_multiplier * IQR
    upper_bound = Q3 + upper_multiplier * IQR
    valid_mask = (residuals >= lower_bound) & (residuals <= upper_bound)
    return y_test[valid_mask], y_pred[valid_mask]

y_test_filtered, y_pred_filtered = filter_outlier_predictions(y_test, y_pred)

# Step 9: Evaluate the model on filtered data
mse_filtered = mean_squared_error(y_test_filtered, y_pred_filtered)
r2_filtered = r2_score(y_test_filtered, y_pred_filtered)
print(f"Filtered Mean Squared Error: {mse_filtered}")
print(f"Filtered R²: {r2_filtered}")

# Step 10: Calculate data loss
unfiltered_count = len(y_test)
filtered_count = len(y_test_filtered)
data_loss_percentage = ((unfiltered_count - filtered_count) / unfiltered_count) * 100
print(f"Number of data points (Unfiltered): {unfiltered_count}")
print(f"Number of data points (Filtered): {filtered_count}")
print(f"Percentage of data lost: {data_loss_percentage:.2f}%")

# Step 11: Plot predicted vs actual values (After Filtering Residuals)
plt.scatter(y_pred_filtered, y_test_filtered, alpha=0.6, color='green')
plt.plot([y_test_filtered.min(), y_test_filtered.max()],
         [y_test_filtered.min(), y_test_filtered.max()],
         'r--', lw=2)
plt.title("Predicted vs Actual (After Filtering Residuals)")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.tight_layout()
plt.show()
