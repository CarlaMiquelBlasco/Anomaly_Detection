import numpy as np
from utils.visualization import *
from config import CONFIG


def run_all_plots(errors, threshold, labels, history, plot_types, test_dataset, autoencoder):
    """
    Calls requested plot functions based on user input.
    """
    if "error_dist" in plot_types:
        print("[PLOT] Plotting error distributions and metrics...")
        plot_results(errors, threshold, labels, history)

    if "event_comparison" in plot_types and CONFIG["MODEL_TYPE"]=="vae":
        print("[PLOT] Plotting event comparison (normal vs anomaly)...")
        anomalies = np.where(errors > threshold)[0]
        original_data = []
        for (inputs, _), _ in test_dataset:
            original_data.append(inputs.numpy())
        original_data = np.concatenate(original_data)
        plot_event_comparison(original_data, anomalies, errors)

    if "latent" in plot_types and CONFIG["MODEL_TYPE"]=="vae":
        print("[PLOT] Plotting latent space projection...")
        max_points = CONFIG["LATENT_PLOT_MAX_POINTS"]
        method = CONFIG["LATENT_PLOT_METHOD"]
        plot_latent(test_dataset, autoencoder, max_points, method)

    if "density" in plot_types and CONFIG["MODEL_TYPE"]=="vae":
        print("[PLOT] Plotting density plots for errors and latent dims...")
        plot_density(test_dataset, autoencoder)

    if "recon_vs_orig" in plot_types:
        print("[PLOT] Plotting original vs reconstructed events...")
        # === Get batch of data ===
        original_vs_reconstructed(test_dataset, autoencoder)
