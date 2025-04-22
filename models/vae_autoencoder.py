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
        """
            Builds the encoder part of the Variational Autoencoder (VAE).

            The encoder transforms input data into a latent representation by applying a sequence of layers.
            It supports three architectural options, selectable by commenting/uncommenting relevant lines:

            - OPTION 1: Conv1D layers (default active here)
                Applies 1D convolutional layers with kernel size 1, followed by BatchNormalization and Dropout.

                Example:
                    x = Conv1D(filters=units, kernel_size=1, activation="relu", kernel_regularizer=l2(l2_strength))(x)

            - OPTION 2: TimeDistributed Dense layers
                Wraps Dense layers in TimeDistributed to apply the same fully connected layer across each time step.

                Example:
                    x = TimeDistributed(Dense(units, activation="relu", kernel_regularizer=l2(l2_strength)))(x)

            - OPTION 3: Flattened input with Dense layers
                Flattens the input and applies standard Dense layers, followed by BatchNormalization and Dropout.

                Example:
                    x = Dense(units, activation="relu", kernel_regularizer=l2(self.l2_strength))(x)

            After the transformation layers, the encoder outputs:
                - `z_mean`: mean of the latent variable distribution
                - `z_log_var`: log variance of the latent variable distribution
                - `z`: sampled latent vector using the reparameterization trick

            Returns:
                tf.keras.Model: A compiled Keras Model representing the encoder.
        """
        inputs = Input(shape=self.input_shape, name="encoder_input") # Input: shape = (700, 3)

        ## Comment/Uncomment for the desired choice Option ##
        # OPTION 1 AND 2:
        x = inputs
        # OPTION 3:
        #x = Flatten()(inputs)

        for units in self.encoder_layers: # Conv1D + BatchNorm + Dropout
            dropout_rate = self.dropout_rate
            l2_strength = self.l2_strength

            ## Comment/Uncomment for the desired choice Option ##
            # OPTION 1:
            x = Conv1D(filters=units, kernel_size=1, activation="relu",kernel_regularizer=l2(l2_strength))(x)
            # OPTION 2:
            #x = TimeDistributed(Dense(units, activation="relu", kernel_regularizer=l2(l2_strength)))(x)
            # OPTION 3:
            #x = Dense(units, activation="relu", kernel_regularizer=l2(self.l2_strength))(x)

            x = BatchNormalization()(x)
            x = Dropout(dropout_rate)(x)

        x = Flatten()(x)
        z_mean = Dense(self.latent_dim, name="z_mean")(x) # mean of the latent distribution
        z_log_var = Dense(self.latent_dim, name="z_log_var")(x) # log variance
        z = Lambda(sampling, output_shape=(self.latent_dim,), name="z")([z_mean, z_log_var]) # sample from latent space using the reparameterization trick

        return Model(inputs, [z_mean, z_log_var, z], name="encoder")

    def _build_decoder(self):
        """
            Builds the decoder part of the Variational Autoencoder (VAE).

            The decoder reconstructs the input data from a sampled latent vector `z`.

            Decoder architecture:
                - Dense layers specified by `decoder_layers`, each followed by BatchNormalization
                - Final Dense layer uses 'tanh' activation (works well with standardized input)
                - Output reshaped to match the original input dimensions

            Notes:
                - Dropout is currently disabled in the decoder for stability, but can be re-enabled as needed.
                - Activation can be adjusted (e.g., 'sigmoid') depending on data preprocessing.

            Returns:
                tf.keras.Model: A compiled Keras Model representing the decoder.
        """
        latent_inputs = Input(shape=(self.latent_dim,), name="decoder_input") # Input: latent vector z
        x = latent_inputs

        for units in self.decoder_layers: # Dense + BatchNorm + Dropout
            x = Dense(units, activation="relu", kernel_regularizer=l2(self.l2_strength))(x)
            x = BatchNormalization()(x)
            #x = Dropout(self.dropout_rate)(x)

        output_dim = self.input_shape[0] * self.input_shape[1]
        x = Dense(output_dim, activation="tanh")(x) # or sigmoid, but since we use standardscaler for pt is better tanh
        outputs = Reshape(self.input_shape)(x)

        return Model(latent_inputs, outputs, name="decoder")

    def call(self, inputs):
        """
            Executes a forward pass through the VAE.

            This method handles both training and inference. It encodes the input into a latent representation,
            samples from the latent space using the reparameterization trick, and reconstructs the input via the decoder.

            Supports optional masking for variable-length inputs.

            Workflow:
                1. Casts input to float32 for compatibility
                2. Encodes input to latent variables (`z_mean`, `z_log_var`) and sample `z`
                3. Reconstructs input from sampled latent vector
                4. Computes total loss, reconstruction loss, and KL divergence loss
                5. Updates internal loss metrics for tracking during training

            Args:
                inputs (Tensor | tuple): Input tensor or (input, mask) tuple

            Returns:
                Tensor: Reconstructed input tensor (same shape as original input)
        """
        if isinstance(inputs, (tuple, list)) and len(inputs) == 2:
            inputs, mask = inputs
        else:
            mask = None

        inputs = tf.cast(inputs, tf.float32) # make sure all inputs are in the correct numerical format for training.
        z_mean, z_log_var, z = self.encoder(inputs) # Encode the input → get z_mean, z_log_var, z.
        reconstructed = self.decoder(z) # Decode z → get reconstructio

        loss, recon_loss, kl_loss = vae_loss(inputs, reconstructed, z_mean, z_log_var, self.beta, mask=mask)
        self.add_loss(loss)


        # Metrics for logging
        self.loss_tracker.update_state(loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)

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

    def get_config(self):
        """
        Returns a serializable config dictionary to enable saving/loading the model.

        Returns:
            dict: Model configuration with parameters
        """
        config = super().get_config()
        config.update({"params": self.params})
        return config

    @classmethod
    def from_config(cls, config):
        """
        Reconstructs the model from a saved config dictionary.

        Args:
            config (dict): Config dictionary (usually from `get_config()`)

        Returns:
            VariationalAutoencoder: Instantiated model
        """
        return cls(params=config["params"])
