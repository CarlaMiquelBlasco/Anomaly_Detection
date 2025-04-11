import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Reshape, Conv1D, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.utils import register_keras_serializable
from tensorflow.keras.layers import Dropout, BatchNormalization, TimeDistributed
from tensorflow.keras.regularizers import l2
from tensorflow.keras.metrics import Mean

from utils.file_utils import sampling, vae_loss

# Registers the class so it can be properly saved and loaded by Keras
@register_keras_serializable(package="CustomModels")
class VariationalAutoencoder(Model):
    """
    Variational Autoencoder (VAE) implementation using convolutional layers.
    Includes KL divergence and reconstruction loss.
    """

    def __init__(self, params, **kwargs):
        super().__init__(**kwargs)

        self.params = params
        self.input_shape = params["input_shape"]
        self.encoder_layers = params["encoder_layers"]
        self.latent_dim = params["latent_dim"]
        self.decoder_layers = params["decoder_layers"]
        self.beta = params["beta"]
        self.dropout_rate = params["dropout_rate"]
        self.l2_strength = params["l2_strength"]

        self.encoder = self._build_encoder()
        self.decoder = self._build_decoder()

        self.loss_tracker = Mean(name="loss")
        self.reconstruction_loss_tracker = Mean(name="reconstruction_loss")
        self.kl_loss_tracker = Mean(name="kl_loss")

    def _build_encoder(self):
        inputs = Input(shape=self.input_shape, name="encoder_input") # Input: shape = (700, 3)
        # OPTION 1 AND 2
        #x = inputs
        # OPTION 3:
        x = Flatten()(inputs)
        for units in self.encoder_layers: # Conv1D + BatchNorm + Dropout
            dropout_rate = self.dropout_rate
            l2_strength = self.l2_strength
            # OPTION 1:
            #x = Conv1D(filters=units, kernel_size=1, activation="relu",kernel_regularizer=l2(l2_strength))(x)
            # OPTION 2:
            #x = TimeDistributed(Dense(units, activation="relu", kernel_regularizer=l2(l2_strength)))(x)
            # OPTION 3:
            x = Dense(units, activation="relu", kernel_regularizer=l2(self.l2_strength))(x)
            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

        x = Flatten()(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(x) # mean of the latent distribution
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x) # log variance
        z = Lambda(sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var]) # sample from latent space using the reparameterization trick

        return Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        latent_inputs = Input(shape=(self.latent_dim,), name="decoder_input") # Input: latent vector z
        x = latent_inputs

        for units in self.decoder_layers: # Dense + BatchNorm + Dropout
            x = Dense(units, activation="relu", kernel_regularizer=l2(self.l2_strength))(x)
            x = BatchNormalization()(x)
            #x = Dropout(self.dropout_rate)(x)

        output_dim = self.input_shape[0] * self.input_shape[1]
        x = Dense(output_dim, activation="sigmoid")(x)
        outputs = Reshape(self.input_shape)(x)

        return Model(latent_inputs, outputs, name="decoder")

    def call(self, inputs):
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            inputs, mask = inputs
            #tf.print("[DEBUG] Mask detected in call().")
            #print("         inputs shape:", tf.shape(inputs))
            #print("         mask shape:  ", tf.shape(mask))
            #print("[DEBUG] Mask values (first sample):", mask[0, :10, 0])
        else:
            #tf.print("[DEBUG] No mask detected in call(). Using default (all ones).")
            #tf.print("         inputs shape:", tf.shape(inputs))
            mask = None

        # debug lr decay:
        #if hasattr(self, 'optimizer'):
        #    try:
        #        tf.print("[DEBUG] Current LR:", self.optimizer.learning_rate)
        #    except Exception as e:
        #        tf.print("[DEBUG] Could not fetch learning rate:", e)

        inputs = tf.cast(inputs, tf.float32) # make sure all inputs are in the correct numerical format for training.
        z_mean, z_log_var, z = self.encoder(inputs) # Encode the input → get z_mean, z_log_var, z.
        reconstructed = self.decoder(z) # Decode z → get reconstructio

        loss, recon_loss, kl_loss = vae_loss(inputs, reconstructed, z_mean, z_log_var, self.beta, mask=mask)
        self.add_loss(loss)


        # === Metrics for logging ===
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        # Add debug print conditionally
        #if training:
        #    tf.print("[DEBUG] z_mean mean/std:", tf.reduce_mean(z_mean), tf.math.reduce_std(z_mean))
        #    tf.print("[DEBUG] z_log_var mean/std:", tf.reduce_mean(z_log_var), tf.math.reduce_std(z_log_var))

        return reconstructed

    @property # Keras uses this to know which metrics to reset each epoch.
    def metrics(self):
        return [
            self.loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker
        ]

    def summary(self):
        """
        Displays summaries of the encoder, decoder, and full model.
        """
        print("\n[INFO] Encoder Summary:")
        self.encoder.summary()

        print("\n[INFO] Decoder Summary:")
        self.decoder.summary()

        print("\n[INFO] VAE Model Summary:")
        self.build((None,) + tuple(self.input_shape))
        super().summary()

    def get_config(self): # saving/loading the model to disk with model.save() or loading with load_model().
        config = super().get_config()
        config.update({"params": self.params})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(params=config["params"])
