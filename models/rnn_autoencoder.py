import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, GRU, Masking, Reshape


class RNNAutoencoder:
    """Builds an RNN-based Autoencoder (LSTM/GRU)."""

    @staticmethod
    def build_model(params):
        input_shape = params["input_shape"]
        rnn_type = params["rnn_type"][0]
        encoder_layers = params["encoder_layers"][0]
        latent_dim = params["latent_dim"]
        activation_encoder = params["activation_encoder"][0]
        activation_decoder = params["activation_decoder"][0]

        input_layer = Input(shape=input_shape)

        # Masking layer for handling variable-length sequences (padded with zeros)
        masked_input = Masking(mask_value=0)(input_layer)

        # Encoder
        encoded = masked_input
        print(params)
        print(encoder_layers)
        for units in encoder_layers:
            if rnn_type == 'LSTM':
                encoded = LSTM(units, activation=activation_encoder, return_sequences=True)(encoded)
            elif rnn_type == 'GRU':
                encoded = GRU(units, activation=activation_encoder, return_sequences=True)(encoded)

        if rnn_type == 'LSTM':
            encoded = LSTM(latent_dim, activation=activation_encoder, return_sequences=False)(encoded)
        elif rnn_type == 'GRU':
            encoded = GRU(latent_dim, activation=activation_encoder, return_sequences=False)(encoded)

        # Decoder
        decoded = Dense(input_shape[0] * input_shape[1], activation=activation_decoder)(encoded)
        decoded = Reshape((input_shape[0], input_shape[1]))(decoded)

        return Model(input_layer, decoded)