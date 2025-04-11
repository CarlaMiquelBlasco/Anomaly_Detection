import datetime
from config import CONFIG
import os
import glob
import tensorflow as tf
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend as K
from tensorflow.keras.utils import register_keras_serializable


def get_model_filename():
    """
    Generates a structured filename based on the current training configuration.
    """
    model_type = CONFIG["MODEL_TYPE"]
    params = CONFIG["AUTOENCODER_PARAMS"][model_type]

    # Extract hyperparameters dynamically
    if model_type == "cnn":
        filters = "-".join(map(str, params["encoder_filters"]))
        kernels = "-".join([f"{k[0]}x{k[1]}" for k in params["kernel_sizes"]])
        latent_dim = params["latent_dim"]
        details = f"filters-{filters}_kernels-{kernels}_latent-{latent_dim}"
    elif model_type == "rnn":
        rnn_type = params["rnn_type"]
        layers = "-".join(map(str, params["encoder_layers"]))
        latent_dim = params["latent_dim"]
        details = f"rnn-{rnn_type}_layers-{layers}_latent-{latent_dim}"
    elif model_type == "vae":
        encoder_layers = "-".join(map(str, params["encoder_layers"]))
        latent_dim = params["latent_dim"]
        details = f"vae_layers-{encoder_layers}_latent-{latent_dim}"
    else:  # MLP
        layers = "-".join(map(str, params["encoder_layers"]))
        details = f"mlp_layers-{layers}"

    epochs = CONFIG["EPOCHS"]
    lr = CONFIG["LEARNING_RATE"]
    timestamp = "07042025" #datetime.datetime.now().strftime("%Y%m%d-%H%M")

    filename = f"{model_type}_{details}_epochs-{epochs}_lr-{lr}_{timestamp}"

    return filename


def get_latest_model():
    """
    Finds the most recently saved model in the specified directory.
    Returns the full file path of the latest saved model.
    """
    model_dir = CONFIG["SAVE_MODEL_DIR"]
    model_type = CONFIG["MODEL_TYPE"]

    # Pattern to match saved models (specific to current model type)
    model_pattern = os.path.join(model_dir, f"{model_type}_*")

    # Get a list of matching models
    model_files = glob.glob(model_pattern)

    if not model_files:
        raise FileNotFoundError(f"[ERROR] No saved models found for model type '{model_type}' in '{model_dir}'")

    # Sort by modification time (latest first)
    model_files.sort(key=os.path.getmtime, reverse=True)

    latest_model_path = model_files[0]  # Get the most recent model

    return latest_model_path


@tf.keras.utils.register_keras_serializable() # Ensures compatibility when saving/loading the model
def custom_loss(y_true, y_pred):
    """
    Custom loss function that computes MSE only for non-zero particles.
    """
    # Create a mask where 1.0 means the particle is not all zeros, otherwise 0.0
    mask = tf.cast(tf.reduce_any(tf.not_equal(y_true, 0), axis=-1, keepdims=True), tf.float32)
    # Compute MSE only for non-zero particles
    loss = MeanSquaredError()(y_true * mask, y_pred * mask)
    return loss

###### VAE UTILS ######

# Register the `sampling` function to enable saving and loading with Keras
@register_keras_serializable(package="CustomModels")
def sampling(args):
    """Applies the reparameterization trick for the Variational Autoencoder (VAE)."""
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim), dtype=tf.float32)
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# VAE loss function defined outside the class for modularity
@tf.keras.utils.register_keras_serializable(package="CustomModels") # Allows the model to be saved/loaded correctly and Makes it serializable within TensorFlow's model saving system
@tf.keras.utils.register_keras_serializable(package="CustomModels")
def vae_loss(inputs, outputs, z_mean, z_log_var, beta, mask=None):
    """
    Computes the VAE loss: weighted reconstruction + KL divergence.
    Applies a mask to ignore padded particles if provided.
    """

    if mask is None:
        print("[DEBUG] No mask provided, using ones")
        mask = tf.ones_like(inputs[..., :1])
    else:
        print("[DEBUG] Mask provided and used")

    # === Feature-wise weights: [pT, eta, phi]
    feature_weights = tf.constant([3.0, 1.0, 1.0], dtype=tf.float32)  # Adjust pT weight here
    feature_weights = tf.reshape(feature_weights, (1, 1, 3))  # Shape: (1, 1, 3) to broadcast

    # === Reconstruction loss (weighted & masked MSE)
    squared_error = tf.square(inputs - outputs)
    weighted_error = squared_error * feature_weights
    masked_error = weighted_error * mask  # mask has shape (batch, 700, 1)

    # Normalize error per sample
    per_event_error = tf.reduce_sum(masked_error, axis=(1, 2)) / (tf.reduce_sum(mask, axis=(1, 2)) + 1e-8)
    reconstruction_loss = tf.reduce_mean(per_event_error)

    # === KL Divergence
    z_log_var = tf.clip_by_value(z_log_var, -10.0, 10.0)
    kl_per_sample = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
    kl_loss = tf.reduce_mean(kl_per_sample)

    total_loss = reconstruction_loss + beta * kl_loss

    return total_loss, reconstruction_loss, kl_loss


def lr_log(epoch, lr):
    """
    Logs the learning rate at the start of each epoch.
    Useful for debugging learning rate schedules.
    """
    new_lr = lr * 0.95  # Decay by 5% every epoch
    print(f"[DEBUG] Epoch {epoch}: LR decayed from {lr:.6f} → {new_lr:.6f}")
    return new_lr