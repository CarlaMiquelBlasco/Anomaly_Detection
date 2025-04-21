import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, GlobalAveragePooling2D, Reshape


class CNNAutoencoder:
    """Builds a CNN-based Autoencoder."""

    @staticmethod
    def build_model(params):
        input_shape = params["input_shape"]
        encoder_filters = params["encoder_filters"]
        kernel_sizes = params["kernel_sizes"]
        latent_dim = params["latent_dim"]

        input_layer = Input(shape=input_shape)

        # Encoder
        encoded = input_layer
        for filters, kernel_size in zip(encoder_filters, kernel_sizes):
            encoded = Conv2D(filters, kernel_size, activation=params["activation_encoder"], padding="same")(encoded)

        encoded = GlobalAveragePooling2D()(encoded)
        encoded = Dense(latent_dim, activation=params["activation_encoder"])(encoded)

        # Decoder
        decoded = Dense(input_shape[0] * input_shape[1], activation=params["activation_decoder"])(encoded)
        decoded = Reshape(input_shape)(decoded)
        decoded = Conv2DTranspose(input_shape[2], (3, 3), activation="sigmoid", padding="same")(decoded)

        return Model(input_layer, decoded)
