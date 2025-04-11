from models.cnn_autoencoder import CNNAutoencoder
from models.rnn_autoencoder import RNNAutoencoder
from models.vae_autoencoder import VariationalAutoencoder


class AutoencoderFactory:
    """Factory class to select the appropriate Autoencoder builder."""

    @staticmethod
    def get_autoencoder(model_type, params):
        if model_type == "cnn":
            return CNNAutoencoder.build_model(params)
        elif model_type == "rnn":
            return RNNAutoencoder.build_model(params)
        elif model_type == "vae":
            return VariationalAutoencoder(params)
        else:
            raise ValueError("Invalid model type. Choose 'cnn', 'rnn', or 'vae'.")



