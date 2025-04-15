import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.utils import custom_object_scope
import hdf5plugin

from config import CONFIG
from data_processing.data_loader import DataLoader
from models.vae_autoencoder import VariationalAutoencoder, sampling

# === Load trained VAE model ===
model_path = CONFIG["MODEL_PATH"]
model_file = os.path.join(model_path, "model.keras")

with custom_object_scope({
    'VariationalAutoencoder': VariationalAutoencoder,
    'sampling': sampling
}):
    vae = tf.keras.models.load_model(model_file)

# === Extract encoder (outputs: z_mean, z_log_var, z) ===
encoder = vae.encoder

# === Load test data ===
data_loader = DataLoader(CONFIG["MODEL_PATH"])
data_loader.prepare_datasets()
test_dataset = data_loader.get_test_dataset()

# === Collect latent representations and labels ===
z_means = []
labels = []

for (inputs, mask), batch_labels in test_dataset:
    inputs = tf.cast(inputs, tf.float32)
    z_mean, _, _ = encoder(inputs, training=False)
    z_means.append(z_mean.numpy())
    labels.append(batch_labels.numpy())

z_means = np.concatenate(z_means)
labels = np.concatenate(labels)

# === Optional: downsample for t-SNE speed ===
max_points = 3000
if len(z_means) > max_points:
    indices = np.random.choice(len(z_means), max_points, replace=False)
    z_means = z_means[indices]
    labels = labels[indices]

# === Choose dimensionality reduction method ===
method = "tsne"  # or "pca"
if method == "tsne":
    reducer = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
elif method == "pca":
    reducer = PCA(n_components=2)
else:
    raise ValueError("Invalid method. Choose 'tsne' or 'pca'.")

z_2d = reducer.fit_transform(z_means)

# === Plot the 2D latent space ===
plt.figure(figsize=(8, 6))
plt.scatter(z_2d[labels == 0, 0], z_2d[labels == 0, 1], c='blue', label='Background', alpha=0.5, s=8)
plt.scatter(z_2d[labels == 1, 0], z_2d[labels == 1, 1], c='red', label='Anomalies', alpha=0.6, s=8)
plt.title(f"Latent Space Visualization ({method.upper()})")
plt.xlabel("Latent Dimension 1")
plt.ylabel("Latent Dimension 2")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# === Histograms of z_mean dimensions ===
import seaborn as sns
for i in range(min(5, z_means.shape[1])):
    sns.histplot(z_means[:, i], kde=True)
    plt.title(f"z_mean[{i}] distribution")
    plt.show()

# === Average z_mean for bg vs anomaly ===
plt.plot(z_means[labels==1].mean(axis=0), label="Anomaly mean z")
plt.plot(z_means[labels==0].mean(axis=0), label="Background mean z")
plt.legend()
plt.title("Average latent representation")
plt.show()

