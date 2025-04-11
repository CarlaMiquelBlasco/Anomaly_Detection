import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import hdf5plugin

from config import CONFIG
from data_processing.data_loader_v2 import DataLoader
from models.vae_autoencoder import VariationalAutoencoder

# === Load trained model ===
model_path = CONFIG["MODEL_PATH"]
model_file = os.path.join(model_path, "model.keras")

with custom_object_scope({'VAEModel': VariationalAutoencoder}):
    model = tf.keras.models.load_model(model_file)

encoder = model.encoder

# === Load test data ===
data_loader = DataLoader(CONFIG["MODEL_PATH"])
data_loader.prepare_datasets()
test_dataset = data_loader.get_test_dataset()

# === Collect latent reps, errors, labels ===
z_means = []
recon_errors = []
labels = []

for (x, mask), label in test_dataset:
    x = tf.cast(x, tf.float32)
    mask = tf.cast(mask, tf.float32)

    # Forward pass
    z_mean, _, _ = encoder(x, training=False)
    x_recon = model((x, mask), training=False)

    # Masked reconstruction error
    squared_error = tf.square(x - x_recon)
    masked_error = squared_error * mask
    error = tf.reduce_sum(masked_error, axis=(1, 2)) / (tf.reduce_sum(mask, axis=(1, 2)) + 1e-8)

    z_means.append(z_mean.numpy())
    recon_errors.append(error.numpy())
    labels.append(label.numpy())

z_means = np.concatenate(z_means, axis=0)
recon_errors = np.concatenate(recon_errors)
labels = np.concatenate(labels)

# === Plot 1: Density plot of reconstruction errors ===
plt.figure(figsize=(8, 5))
sns.kdeplot(recon_errors[labels == 0], label="Background", fill=True, color="blue")
sns.kdeplot(recon_errors[labels == 1], label="Anomaly", fill=True, color="red")
plt.xlabel("Reconstruction Error")
plt.ylabel("Density")
plt.title("Density Plot of Reconstruction Errors")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Plot 2: Density plots of first 5 latent dimensions ===
for i in range(min(5, z_means.shape[1])):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(z_means[labels == 0, i], label="Background", fill=True)
    sns.kdeplot(z_means[labels == 1, i], label="Anomaly", fill=True)
    plt.xlabel(f"z_mean[{i}]")
    plt.title(f"Density Plot of Latent Dimension {i}")
    plt.legend()
    plt.tight_layout()
    plt.show()
