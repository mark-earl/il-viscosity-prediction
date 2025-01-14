import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

# Load the data from the Excel file
data_path = 'RDKit/data/heavyweight-ils.xlsx'
df = pd.read_excel(data_path)

# Drop unnecessary columns
excluded_il_col = df["Excluded IL"]
df = df.drop(columns=[
    'IL ID', 'Cation', 'Anion',
    'cation_Ion type', 'cation_Family', 'cation_Number of ILs composed of the ion',
    'anion_Ion type', 'anion_Family', 'anion_Number of ILs composed of the ion'
])

# Filter data based on the "Excluded IL" attribute
included_data = df[excluded_il_col == False]
excluded_data = df[excluded_il_col == True]

# Identify and filter out outliers in the target variable ('η / mPa s')
def filter_outliers(data, column, lower_multiplier=1.5, upper_multiplier=1.5):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - lower_multiplier * IQR
    upper_bound = Q3 + upper_multiplier * IQR
    return data[(data[column] >= lower_bound) & (data[column] <= upper_bound)], len(data), len(data[(data[column] >= lower_bound) & (data[column] <= upper_bound)])

# Apply initial filtering on target variable
included_data, total_before_filter, total_after_filter = filter_outliers(included_data, 'η / mPa s')
data_loss_initial = ((total_before_filter - total_after_filter) / total_before_filter) * 100

# Define features and target variable for included data
X_included = included_data.drop(columns=['η / mPa s'])  # Features
y_included = included_data['η / mPa s']  # Target

# Run the model 50 times and collect R² values
num_runs = 50
r2_scores = []

for i in range(num_runs):
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=i)

    # Train the CatBoost model
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.1,
        depth=7,
        verbose=0  # Suppress output for multiple runs
    )
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    r2_scores.append(r2)

# Calculate 95% confidence interval for R²
mean_r2 = np.mean(r2_scores)
std_r2 = np.std(r2_scores, ddof=1)  # Sample standard deviation
confidence_interval = norm.interval(0.95, loc=mean_r2, scale=std_r2 / np.sqrt(num_runs))

# Print confidence interval
print(f"95% Confidence Interval for R²: {confidence_interval}")

# Plot results from a random run
X_train, X_test, y_train, y_test = train_test_split(X_included, y_included, test_size=0.2, random_state=0)
model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.1,
    depth=7,
    verbose=200
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(y_pred, y_test, alpha=0.6, color='blue', label=f"Included ILs (R²: {r2_scores[0]:.2f})")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Ideal Fit")
plt.title("Predicted vs Actual Values (First Run)")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Sort R² scores for a smooth line plot
plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")

# Highlight the confidence interval
plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3, label="95% CI")

# Plot mean R² as a horizontal line
plt.axhline(mean_r2, color='green', linestyle='-', label="Mean R²")

# Labels and title
plt.title("R² Scores Across 50 Runs with 95% Confidence Interval")
plt.xlabel("Run Index")
plt.ylabel("R² Values")
plt.legend()
plt.tight_layout()
plt.show()
