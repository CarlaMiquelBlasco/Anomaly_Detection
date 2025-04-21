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
        '''
        for (batch, mask), labels in test_dataset:
            batch_np = batch.numpy()
            mask_np = mask.numpy()
            labels_np = labels.numpy()
            break  # Just grab the first batch

        # === Select background and anomaly indices ===
        bg_idx = np.where(labels_np == 0)[0][0]
        anomaly_idx = np.where(labels_np == 1)[0][0]

        original_bg = batch_np[bg_idx]
        mask_bg = mask_np[bg_idx]
        original_anomaly = batch_np[anomaly_idx]
        mask_anomaly = mask_np[anomaly_idx]

        # === Reconstruct ===
        reconstructed_bg = autoencoder((tf.expand_dims(original_bg, 0), tf.expand_dims(mask_bg, 0)), training=False).numpy()[0]
        reconstructed_anomaly = autoencoder((tf.expand_dims(original_anomaly, 0), tf.expand_dims(mask_anomaly, 0)), training=False).numpy()[0]

        plot_event(original_bg, reconstructed_bg, mask_bg, "Background Event: Original vs Reconstructed")
        plot_event(original_anomaly, reconstructed_anomaly, mask_anomaly, "Anomaly Event: Original vs Reconstructed")

    if not plot_types:
        print("[INFO] No plots selected. Use --plots to specify which to generate.")
    '''

