import matplotlib.pyplot as plt
import numpy as np

# Plot data
def plot_results(included_data, excluded_data, y_test, y_pred, r2_rand):

    included_viscosities = included_data['Reference Viscosity Log']
    excluded_viscosities = excluded_data['Reference Viscosity Log']

    y_test_values = y_test
    y_pred_values = y_pred

    fig = plt.figure(figsize=(12,8))

    # Plot excluded data (gray)
    plt.scatter(
        excluded_viscosities,
        excluded_viscosities,
        alpha=0.6, color='gray', label="Excluded ILs"
    )

    # Plot included data (blue)
    plt.scatter(
        included_viscosities,
        included_viscosities,
        alpha=0.6, color='blue', label="Included ILs"
    )

    # Plot test data (red)
    plt.scatter(
        y_pred_values, y_test_values,
        alpha=0.8, color='red', label=f"Test Data (R²: {r2_rand:.2f})"
    )

    # Add ideal fit line
    plt.plot(
        [included_viscosities.min(), included_viscosities.max()],
        [included_viscosities.min(), included_viscosities.max()],
        'r--', lw=2, label="Ideal Fit"
    )

    # Customize plot
    plt.title("Log-Scale Predicted vs Actual Viscosities")
    plt.xlabel("Predicted Log Viscosity (log10[mPa s])")
    plt.ylabel("Actual Log Viscosity (log10[mPa s])")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return fig
