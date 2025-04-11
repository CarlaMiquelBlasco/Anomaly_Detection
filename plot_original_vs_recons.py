import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import hdf5plugin

from config import CONFIG
from data_processing.data_loader_v2 import DataLoader
from models.vae_autoencoder import VariationalAutoencoder, sampling
from tensorflow.keras.utils import custom_object_scope

# === Load model ===
model_path = os.path.join(CONFIG["MODEL_PATH"], "model.keras")

with custom_object_scope({'VariationalAutoencoder': VariationalAutoencoder, 'sampling': sampling}):
    model = tf.keras.models.load_model(model_path)

# === Load test data ===
data_loader = DataLoader(CONFIG["MODEL_PATH"])
data_loader.prepare_datasets()
test_dataset = data_loader.get_test_dataset()

# === Get batch of data ===
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
reconstructed_bg = model((tf.expand_dims(original_bg, 0), tf.expand_dims(mask_bg, 0)), training=False).numpy()[0]
reconstructed_anomaly = model((tf.expand_dims(original_anomaly, 0), tf.expand_dims(mask_anomaly, 0)), training=False).numpy()[0]

# === Plotting ===
def plot_event(original, reconstructed, mask, title):
    mask_flat = mask.squeeze()
    active_particles = original[mask_flat.squeeze() > 0]
    recon_particles = reconstructed[mask_flat.squeeze() > 0]

    num_to_plot = min(50, active_particles.shape[0])
    x = np.arange(num_to_plot)

    fig, axs = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    feature_names = ['pT', 'η', 'φ']
    for i in range(3):
        axs[i].plot(x, active_particles[:num_to_plot, i], label="Original", linewidth=2)
        axs[i].plot(x, recon_particles[:num_to_plot, i], label="Reconstructed", linestyle='--')
        axs[i].set_ylabel(feature_names[i])
        axs[i].grid(True)
        axs[i].legend()

    axs[-1].set_xlabel("Particle Index")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

plot_event(original_bg, reconstructed_bg, mask_bg, "Background Event: Original vs Reconstructed")
plot_event(original_anomaly, reconstructed_anomaly, mask_anomaly, "Anomaly Event: Original vs Reconstructed")
