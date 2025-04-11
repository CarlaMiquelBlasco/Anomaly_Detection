import os
import pickle
import numpy as np
import tensorflow as tf

from config import CONFIG
from utils.file_utils import custom_loss
from models.vae_autoencoder import VariationalAutoencoder, sampling
from tensorflow.keras.utils import custom_object_scope
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix


class AutoencoderTester:
    """
    Loads a trained autoencoder and evaluates it on unseen test data.
    """

    def __init__(self, model, model_path):
        self.model_path = os.path.join(CONFIG["MODEL_PATH"], "model.keras")

        self.model_type = CONFIG["MODEL_TYPE"]#self._extract_model_type(self.model_path)
        print(f"[INFO] Detected model type: {self.model_type}")

        if self.model_type == "vae":
            with custom_object_scope({
                "VariationalAutoencoder": VariationalAutoencoder
            }):
                self.model = tf.keras.models.load_model(self.model_path)

        else:
            self.model = tf.keras.models.load_model(self.model_path, custom_objects={"custom_loss": custom_loss})

    def _extract_model_type(self, model_path):
        """
        Infers model type (cnn, rnn, vae) from the model directory name.
        """
        name = os.path.basename(CONFIG["MODEL_PATH"]).lower()
        if name.startswith("cnn"):
            return "cnn"
        elif name.startswith("rnn"):
            return "rnn"
        elif name.startswith("vae"):
            return "vae"
        else:
            raise ValueError(f"[ERROR] Could not determine model type from: {name}")

    def test(self, test_dataset):
        """
        Evaluates the autoencoder on the test dataset.
        Computes reconstruction error, predictions, and metrics.

        Args:
            test_dataset: tf.data.Dataset yielding (features, labels)

        Returns:
            errors: array of reconstruction errors
            threshold: float, decision threshold
            predictions: binary array (0 = normal, 1 = anomaly)
            history: training history loaded from file
            true_labels: true class labels from the test set
        """
        print("[INFO] Evaluating model on test dataset...")

        all_errors = []
        all_labels = []

        for (inputs, mask), labels in test_dataset:
            inputs = tf.cast(inputs, tf.float32)
            mask = tf.cast(mask, tf.float32)

            if self.model_type == "cnn":
                inputs = tf.expand_dims(inputs, axis=-1)

            # Passes the input inputs through the model to get the reconstruction
            reconstructed = self.model((inputs, mask), training=False)
            reconstructed = tf.cast(reconstructed, tf.float32)

            # Handle any NaNs in output
            if tf.reduce_any(tf.math.is_nan(reconstructed)):
                print("[WARNING] NaNs in reconstruction. Replacing with zeros.")
                reconstructed = tf.where(tf.math.is_nan(reconstructed), tf.zeros_like(reconstructed), reconstructed)

            # Compute reconstruction error
            if self.model_type == "cnn":
                errors = tf.reduce_mean(tf.square(inputs - reconstructed), axis=(1, 2, 3))
            elif self.model_type=="rnn":
                errors = tf.reduce_mean(tf.square(inputs - reconstructed), axis=(1, 2))
            else:
                # Mask: only include non-zero particles
                mask = tf.cast(tf.reduce_any(inputs > 0.01, axis=-1, keepdims=True), tf.float32)

                squared_error = tf.square(inputs - reconstructed)
                masked_error = squared_error * mask
                errors = tf.reduce_sum(masked_error, axis=(1, 2)) / (tf.reduce_sum(mask, axis=(1, 2)) + 1e-8)

            if tf.reduce_any(tf.math.is_nan(errors)):
                print("[WARNING] NaNs in error. Replacing with inf.")
                errors = tf.where(tf.math.is_nan(errors), tf.constant(np.inf, dtype=errors.dtype), errors)

            all_errors.append(errors.numpy())
            all_labels.append(labels.numpy())

        errors = np.concatenate(all_errors)
        true_labels = np.concatenate(all_labels)

        # Load or compute threshold
        threshold_path = os.path.join(os.path.dirname(self.model_path), "threshold.txt")
        if os.path.exists(threshold_path):
            with open(threshold_path, "r") as f:
                threshold = float(f.read().strip())
            print(f"[INFO] Loaded threshold from file: {threshold}")
        else:
            threshold = np.percentile(errors, CONFIG["THRESHOLD_PERCENTILE"])
            print(f"[INFO] No threshold file found. Using percentile: {threshold:.4f}")

        # Load training history
        history_path = os.path.join(os.path.dirname(self.model_path), "history.pkl")
        if os.path.exists(history_path):
            with open(history_path, "rb") as f:
                history = pickle.load(f)
        else:
            history = None
            print(f"[INFO] No history file found.")

        # Make predictions and evaluate
        predictions = (errors > threshold).astype(int)
        accuracy = np.mean(predictions == true_labels)

        print(f"[INFO] Test Accuracy: {accuracy:.4f}")
        print("[INFO] Confusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
        print(f"Precision: {precision_score(true_labels, predictions):.4f}")
        print(f"Recall:    {recall_score(true_labels, predictions):.4f}")
        print(f"F1 Score:  {f1_score(true_labels, predictions):.4f}")
        print(
            f"[DEBUG] Reconstruction error stats: mean={errors.mean():.4f}, std={errors.std():.4f}, min={errors.min():.4f}, max={errors.max():.4f}")
        print(f"[DEBUG] errors.shape: {errors.shape}, labels.shape: {true_labels.shape}")

        return errors, threshold, predictions, history, true_labels

