import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import tensorflow as tf
import os
import datetime

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

    # ✅ Plot distribution
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

     # 🔹 Ensure they are 1D
     background_errors = background_errors.flatten()
     anomaly_errors = anomaly_errors.flatten()

     # ✅ Fix: Provide `color` as a list
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

     #plt.plot(history["loss"], label="Train Loss")
     #plt.plot(history["val_loss"], label="Val Loss")
     #plt.xlabel("Epoch")
     #plt.ylabel("Loss")
     #plt.title("Training vs Validation Loss")
     #plt.legend()
     #plt.show()

