import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
import os
import datetime
from config import CONFIG

def setup_tensorboard(log_dir="logs/fit/"):
    """
    Initializes a TensorBoard callback for logging training metrics.
    """
    log_dir = log_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

def plot_training_history(history):
    """
    Plots training and validation loss over epochs.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'], label="Train Loss")
    plt.plot(history.history['val_loss'], label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Training Loss over Epochs")
    plt.show()

def log_optuna_study(study, log_dir="logs/optuna"):
    """
    Logs Optuna hyperparameter tuning results to TensorBoard.
    """
    writer = tf.summary.create_file_writer(log_dir)
    with writer.as_default():
        for i, trial in enumerate(study.trials):
            tf.summary.scalar("Optuna/Best Loss", trial.value, step=i)
            for param_name, param_value in trial.params.items():
                if isinstance(param_value, (int, float)):
                    tf.summary.scalar(f"Optuna/{param_name}", param_value, step=i)
                else:
                    # Log non-scalar values as text
                    tf.summary.text(f"Optuna/{param_name}", str(param_value), step=i)
    writer.close()

def analyze_distribution(labels, dataset_name="Dataset"):
    """
    Analyzes and visualizes the distribution of background (0) vs. signal (1) events
    in a given dataset (validation or test).

    Parameters:
    - labels: NumPy array of event labels (0=background, 1=signal).
    - dataset_name: Name of the dataset for better visualization.
    """
    print(f"[INFO] Analyzing distribution for {dataset_name}...")

    # Count occurrences of each label
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))

    total_events = sum(counts)
    background_percentage = (distribution[0] / total_events) * 100
    signal_percentage = (distribution[1] / total_events) * 100

    print(f"[INFO] {dataset_name} background events: {background_percentage:.2f}% ({distribution[0]})")
    print(f"[INFO] {dataset_name} signal events: {signal_percentage:.2f}% ({distribution[1]})")

    # Plot distribution
    plt.figure(figsize=(6, 4))
    sns.barplot(x=unique, y=counts, palette=["blue", "red"])
    plt.xlabel("Event Type (0 = Background, 1 = Signal)")
    plt.ylabel("Number of Events")
    plt.title(f"Distribution of Background vs Signal in {dataset_name}")
    plt.xticks(ticks=[0, 1], labels=["Background", "Signal"])
    plt.show()

def plot_reconstruction_error_distribution(errors, threshold):
    """
    Plots the reconstruction error distribution and the anomaly detection threshold.
    :param errors: Reconstruction error for each event.
    :param threshold: Anomaly detection threshold.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=50, alpha=0.6, color='blue', label="Reconstruction Errors")
    plt.axvline(threshold, color='red', linestyle='dashed', linewidth=2, label="Anomaly Threshold")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Number of Events")
    plt.legend()
    plt.title("Reconstruction Error Distribution")
    plt.show()


def plot_event_comparison(original_event, anomalies, errors):
    """
    Plots an example of a normal event vs. an anomalous event.
    :param original_event: The event data.
    :param anomalies: Indices of detected anomalies.
    :param errors: Reconstruction errors of all events.
    """
    normal_event_idx = np.argmin(errors)  # Event with lowest error
    anomaly_idx = anomalies[0]  # Take first detected anomaly

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(original_event[normal_event_idx].reshape(700, 3), cmap="coolwarm")
    plt.title("Normal Event")

    plt.subplot(1, 2, 2)
    plt.imshow(original_event[anomaly_idx].reshape(700, 3), cmap="coolwarm")
    plt.title("Anomalous Event")

    plt.show()

def plot_results(errors, threshold, labels, history):
     """
     Plots reconstruction error distribution, Precision-Recall, and ROC curves.
     """
     # Ensure inputs are NumPy arrays
     errors = np.array(errors)
     labels = np.array(labels)

     # Split errors based on true labels
     background_errors = errors[labels == 0]
     anomaly_errors = errors[labels == 1]

     # Ensure they are 1D
     background_errors = background_errors.flatten()
     anomaly_errors = anomaly_errors.flatten()

     plt.figure(figsize=(10, 5))
     plt.hist([background_errors, anomaly_errors], bins=50, alpha=0.6, color=["blue", "red"], label=["Background", "Anomalies"])
     plt.axvline(threshold, color='black', linestyle='dashed', linewidth=1, label="Threshold")
     plt.xlabel("Reconstruction Error")
     plt.ylabel("Frequency")
     plt.title("Reconstruction Error Distribution")
     plt.legend()
     plt.show()

     # Compute Precision-Recall and ROC curves
     precision, recall, _ = precision_recall_curve(labels, errors)
     fpr, tpr, _ = roc_curve(labels, errors)
     roc_auc = auc(fpr, tpr)

     # Plot Precision-Recall Curve
     plt.figure(figsize=(8, 6))
     plt.plot(recall, precision, marker='.', label="Precision-Recall Curve")
     plt.xlabel("Recall")
     plt.ylabel("Precision")
     plt.title("Precision-Recall Curve")
     plt.legend()
     plt.show()

     # Plot ROC Curve
     plt.figure(figsize=(8, 6))
     plt.plot(fpr, tpr, marker='.', label=f"ROC Curve (AUC = {roc_auc:.2f})")
     plt.plot([0, 1], [0, 1], linestyle='--', color='gray')  # Random classifier line
     plt.xlabel("False Positive Rate")
     plt.ylabel("True Positive Rate")
     plt.title("Receiver Operating Characteristic (ROC) Curve")
     plt.legend()
     plt.show()

     # Scatter Plot: Reconstruction Error vs. True Labels
     plt.figure(figsize=(10, 5))
     plt.scatter(range(len(errors)), errors, c=labels, cmap="coolwarm", alpha=0.7)
     plt.axhline(threshold, color='black', linestyle='dashed', linewidth=1, label="Threshold")
     plt.xlabel("Event Index")
     plt.ylabel("Reconstruction Error")
     plt.title("Reconstruction Error vs. True Labels")
     plt.legend()
     plt.show()

    # plot Loss history ig history is available
     try:
         plt.plot(history["loss"], label="Train Loss")
         plt.plot(history["val_loss"], label="Val Loss")
         plt.xlabel("Epoch")
         plt.ylabel("Loss")
         plt.title("Training vs Validation Loss")
         plt.legend()
         plt.show()
     except Exception as e:
         print(f"[WARNING] Could not plot training/validation loss: {e}")



def plot_latent(test_dataset, autoencoder, max_points, method):
    # === Collect latent representations and labels ===
    z_means = []
    labels = []

    for (inputs, mask), batch_labels in test_dataset:
        inputs = tf.cast(inputs, tf.float32)
        z_mean, _, _ = autoencoder.encoder(inputs, training=False)
        z_means.append(z_mean.numpy())
        labels.append(batch_labels.numpy())

    z_means = np.concatenate(z_means)
    labels = np.concatenate(labels)

    # === Optional: downsample for t-SNE speed ===
    if len(z_means) > max_points:
        indices = np.random.choice(len(z_means), max_points, replace=False)
        z_means = z_means[indices]
        labels = labels[indices]

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
    plt.plot(z_means[labels == 1].mean(axis=0), label="Anomaly mean z")
    plt.plot(z_means[labels == 0].mean(axis=0), label="Background mean z")
    plt.legend()
    plt.title("Average latent representation")
    plt.show()


def plot_density(test_dataset,model):
    # === Collect latent reps, errors, labels ===
    z_means = []
    recon_errors = []
    labels = []

    for (x, mask), label in test_dataset:
        x = tf.cast(x, tf.float32)
        mask = tf.cast(mask, tf.float32)

        # Forward pass
        z_mean, _, _ = model.encoder(x, training=False)
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


def original_vs_reconstructed(test_dataset, model):
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
    def reconstruct(input_data, input_mask):
        input_data = tf.expand_dims(input_data, 0)
        input_mask = tf.expand_dims(input_mask, 0)
        if CONFIG["MODEL_TYPE"] == "vae":
            return model((input_data, input_mask), training=False).numpy()[0]
        else:
            return model(input_data, training=False).numpy()[0]

    reconstructed_bg = reconstruct(original_bg, mask_bg)
    reconstructed_anomaly = reconstruct(original_anomaly, mask_anomaly)

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


