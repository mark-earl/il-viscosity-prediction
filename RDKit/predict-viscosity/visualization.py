import matplotlib.pyplot as plt
import numpy as np

# Plot data
def plot_results(included_data, excluded_data, y_test, y_pred, r2_rand):
    included_log_viscosities = np.log10(included_data['Reference Viscosity'])
    excluded_log_viscosities = np.log10(excluded_data['Reference Viscosity'])
    y_test_log = np.log10(y_test)
    y_pred_log = np.log10(y_pred)

    plt.figure(figsize=(12, 8))

    # Plot excluded data (gray)
    plt.scatter(
        np.log10(excluded_data['Reference Viscosity']),
        np.log10(excluded_data['Reference Viscosity']),
        alpha=0.6, color='gray', label="Excluded ILs"
    )

    # Plot included data (blue)
    plt.scatter(
        included_log_viscosities,
        included_log_viscosities,
        alpha=0.6, color='blue', label="Included ILs"
    )

    # Plot test data (red)
    plt.scatter(
        y_pred_log, y_test_log,
        alpha=0.8, color='red', label=f"Test Data (R²: {r2_rand:.2f})"
    )

    # Add ideal fit line
    plt.plot(
        [included_log_viscosities.min(), included_log_viscosities.max()],
        [included_log_viscosities.min(), included_log_viscosities.max()],
        'r--', lw=2, label="Ideal Fit"
    )

    # Customize plot
    plt.title("Log-Scale Predicted vs Actual Viscosities")
    plt.xlabel("Predicted Log Viscosity (log10[mPa s])")
    plt.ylabel("Actual Log Viscosity (log10[mPa s])")
    plt.legend()
    plt.tight_layout()
    plt.show()
