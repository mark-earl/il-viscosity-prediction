import io
import matplotlib.pyplot as plt
import streamlit as st

def plot_results(included_data, excluded_data, y_test, y_pred, r2_rand):
    """Generates and allows download of a plot for log-scale predicted vs actual viscosities."""
    included_viscosities = included_data['Reference Viscosity Log']
    excluded_viscosities = excluded_data['Reference Viscosity Log']

    y_test_values = y_test
    y_pred_values = y_pred

    fig = plt.figure(figsize=(12, 8))

    plt.scatter(excluded_viscosities, excluded_viscosities, alpha=0.6, color='gray', label="Excluded ILs")
    plt.scatter(included_viscosities, included_viscosities, alpha=0.6, color='blue', label="Included ILs")
    plt.scatter(y_pred_values, y_test_values, alpha=0.8, color='red', label=f"Test Data (R²: {r2_rand:.2f})")

    plt.plot(
        [included_viscosities.min(), included_viscosities.max()],
        [included_viscosities.min(), included_viscosities.max()],
        'r--', lw=2, label="Ideal Fit"
    )

    plt.title("Log-Scale Predicted vs Actual Viscosities")
    plt.xlabel("Predicted Log Viscosity (log10[mPa s])")
    plt.ylabel("Actual Log Viscosity (log10[mPa s])")
    plt.legend()
    plt.tight_layout()

    st.pyplot(fig)

    # Save plot as PNG and allow download
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download Plot as PNG",
        data=buffer,
        file_name="viscosity_results_plot.png",
        mime="image/png"
    )


def plot_confidence_interval(r2_scores, confidence_interval, mean_r2, title_suffix=""):
    """Generates and allows download of a plot for R² scores with a confidence interval."""
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

    st.pyplot(fig)

    # Save plot as PNG and allow download
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)

    st.download_button(
        label="Download R² Plot as PNG",
        data=buffer,
        file_name="confidence_interval_plot.png",
        mime="image/png"
    )
