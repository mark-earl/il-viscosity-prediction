import matplotlib.pyplot as plt

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

def plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix=""):
    """Plots R² scores with a confidence interval."""
    fig = plt.figure()
    plt.plot(r2_scores, marker='o', linestyle='-', color='blue', label="R² Scores")
    plt.axhspan(confidence_interval[0], confidence_interval[1], color='yellow', alpha=0.3,
                 label=f"95% CI: ({confidence_interval[0]:.2f}, {confidence_interval[1]:.2f})")
    plt.axhline(mean_r2, color='green', linestyle='-', label=f"Mean R²: {mean_r2:.2f}")

    plt.title(f"R² Scores Across Runs with 95% Confidence Interval {title_suffix}")
    plt.xlabel("Run Index")
    plt.ylabel("R² Values")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return fig
