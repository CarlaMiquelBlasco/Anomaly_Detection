import time
import numpy as np
import tensorflow as tf
from config import CONFIG
from utils.file_utils import *
from utils.visualization import setup_tensorboard
from optuna.integration import TFKerasPruningCallback
from utils.beta_scheduler import BetaWarmupScheduler
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.callbacks import LearningRateScheduler



class AutoencoderTrainer:
    """
    Handles training and evaluation of an autoencoder model for anomaly detection.
    """

    def __init__(self, model, learning_rate):
        self.model = model
        self.model_type = CONFIG["MODEL_TYPE"]
        self.learning_rate = learning_rate if learning_rate else CONFIG["LEARNING_RATE"]

    def train(self, train_dataset, val_dataset, steps_per_epoch, trial=None):
        """
        Trains the model on background-only data and monitors validation loss.
        Applies early stopping and saves the best model overall.

        Parameters:
        - train_dataset: tf.data.Dataset
        - val_dataset: tf.data.Dataset
        - steps_per_epoch: int, number of batches per epoch
        - trial: Optional Optuna trial
        """
        print(f"[INFO] Starting training for model type: {self.model_type.upper()}")

        start_time = time.time()

        # Compile model
        # Define decaying learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=self.learning_rate,
            decay_steps=100000,  # Decay after N steps
            decay_rate=0.96,  # Multiply LR by this factor
            staircase=True  # If True: discrete steps; False: smooth decay
        )

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=1.0)
        if self.model_type == "vae":
            self.model.compile(optimizer=optimizer)
        elif self.model_type == "cnn":
            self.model.compile(optimizer=optimizer, loss=custom_loss)
        else:
            self.model.compile(optimizer=optimizer, loss="mse")

        # Callbacks
        callbacks = [setup_tensorboard()]

        # Checkpoint (global best)
        checkpoint_path = os.path.join(CONFIG["MODEL_PATH"], "model.keras")

        checkpoint_cb = ModelCheckpoint(
            checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
            mode="min"
        )
        callbacks.append(checkpoint_cb)

        # Early stopping
        early_stop_cb = EarlyStopping(
            monitor="val_loss",
            patience=CONFIG.get("PATIENCE", 5),
            restore_best_weights=True,
            verbose=1,
            mode = "min"
        )
        callbacks.append(early_stop_cb)

        if self.model_type=="vae":
            # Increasing beta gradually
            beta_cb = BetaWarmupScheduler(self.model, max_beta=self.model.beta, warmup_epochs=5)
            callbacks.append(beta_cb)

        # Optuna pruning (optional)
        if trial:
            callbacks.append(TFKerasPruningCallback(trial, 'val_loss'))

        # Train model
        if self.model_type == "vae":
            wrapped_train_dataset = train_dataset.map(lambda x, mask: ((x, mask), x))
            wrapped_val_dataset = val_dataset.map(lambda x, mask: ((x, mask), x))
        else:
            wrapped_train_dataset = train_dataset.map(lambda x: (x, x))
            wrapped_val_dataset = val_dataset.map(lambda x: (x, x))

        history = self.model.fit(
            wrapped_train_dataset,
            validation_data=wrapped_val_dataset,
            epochs=CONFIG["EPOCHS"],
            steps_per_epoch=steps_per_epoch,
            callbacks=callbacks
        )

        # Check for NaNs in training loss
        losses = history.history.get("loss", [])
        if any(np.isnan(loss) for loss in losses):
            print("[ERROR - DEBUG] Detected NaN in training loss!")

        elapsed = time.time() - start_time
        print(f"[INFO] Training completed in {time.strftime('%H:%M:%S', time.gmtime(elapsed))}")

        # Compute and save threshold on validation set
        errors_val = []
        for x_val, mask_val in val_dataset:
            x_val = tf.cast(x_val, tf.float32)
            mask_val = tf.cast(mask_val, tf.float32)

            z_mean, z_log_var, _ = self.model.encoder(x_val, training=False)
            reconstructed = self.model.decoder(z_mean, training=False)

            per_event_errors = vae_loss(x_val, reconstructed, z_mean, z_log_var,
                                        beta=self.model.beta, mask=mask_val,
                                        return_only_errors=True)

            errors_val.append(per_event_errors.numpy())

        errors_val = np.concatenate(errors_val)
        threshold = np.percentile(errors_val, CONFIG["THRESHOLD_PERCENTILE"])

        return history, threshold