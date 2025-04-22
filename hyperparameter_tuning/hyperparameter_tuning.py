import os
import pickle
import optuna
import tensorflow as tf
from config import CONFIG
from data_processing.data_loader import DataLoader
from models.autoencoders_factory import AutoencoderFactory
from models.train import AutoencoderTrainer


def objective(trial):
    """
    Optuna objective function for hyperparameter tuning.
    Trains an autoencoder using chunked data and returns validation loss.
    """
    model_type = CONFIG["MODEL_TYPE"]
    params = dict(CONFIG["AUTOENCODER_PARAMS"][model_type])  # Clone base params
    use_optuna = CONFIG["USE_OPTUNA"]

    # Load and prepare datasets
    print("[INFO] Preparing datasets with streaming...")
    data_loader = DataLoader(CONFIG["MODEL_PATH"])
    data_loader.prepare_datasets()

    train_dataset = data_loader.get_lazy_train_dataset() #(data, mask)
    val_dataset = data_loader.get_validation_dataset() #(data, mask)

    train_size = len(data_loader.train_indices)
    steps_per_epoch = train_size // CONFIG["BATCH_SIZE"]
    print(f"[INFO] Training steps per epoch: {steps_per_epoch}")

    # Suggest hyperparameters with Optuna
    if use_optuna:
        if model_type == "vae":
            params["encoder_layers"] = trial.suggest_categorical("encoder_layers", params["encoder_layers"])
            params["latent_dim"] = trial.suggest_int("latent_dim", params["latent_dim_min"], params["latent_dim_max"], step=params["latent_dim_step"])
            params["decoder_layers"] = trial.suggest_categorical("decoder_layers", params["decoder_layers"])
            params["beta"] = trial.suggest_float("beta", params["beta_min"], params["beta_max"])

        elif model_type == "cnn":
            train_dataset = train_dataset.map(lambda x, m: x)
            val_dataset = val_dataset.map(lambda x, m: x)
            params["encoder_filters"] = trial.suggest_categorical("encoder_filters", params["encoder_filters"])
            params["kernel_sizes"] = trial.suggest_categorical("kernel_sizes", params["kernel_sizes"])
            params["latent_dim"] = trial.suggest_int("latent_dim", params["latent_dim_min"], params["latent_dim_max"], step=params["latent_dim_step"])
            params["activation_encoder"] = trial.suggest_categorical("activation_encoder", params["activation_encoder"])
            params["activation_decoder"] = trial.suggest_categorical("activation_decoder", params["activation_decoder"])

        elif model_type == "rnn":
            train_dataset = train_dataset.map(lambda x, m: x)
            val_dataset = val_dataset.map(lambda x, m: x)
            params["encoder_layers"] = trial.suggest_categorical("encoder_layers", params["encoder_layers"])
            params["latent_dim"] = trial.suggest_int("latent_dim", params["latent_dim_min"], params["latent_dim_max"], step=params["latent_dim_step"])
            params["rnn_type"] = trial.suggest_categorical("rnn_type", params["rnn_type"])
            params["activation_encoder"] = trial.suggest_categorical("activation_encoder", params["activation_encoder"])
            params["activation_decoder"] = trial.suggest_categorical("activation_decoder", params["activation_decoder"])

        learning_rate = trial.suggest_float("learning_rate", CONFIG["LEARNING_RATE_MIN"], CONFIG["LEARNING_RATE_MAX"], log=True)
    else:
        # Use defaults if Optuna is disabled
        if model_type == "vae":
            pass  # All defaults are already loaded
        elif model_type == "cnn":
            train_dataset = train_dataset.map(lambda x, m: x)
            val_dataset = val_dataset.map(lambda x, m: x)
            params["encoder_filters"] = params["encoder_filters"]
            params["kernel_sizes"] = params["kernel_sizes"]
            params["activation_encoder"] = params["activation_encoder"]
            params["activation_decoder"] = params["activation_decoder"]
        elif model_type == "rnn":
            train_dataset = train_dataset.map(lambda x, m: x)
            val_dataset = val_dataset.map(lambda x, m: x)
            params["encoder_layers"] = params["encoder_layers"]
            params["rnn_type"] = params["rnn_type"]
            params["activation_encoder"] = params["activation_encoder"]
            params["activation_decoder"] = params["activation_decoder"]

        learning_rate = CONFIG["LEARNING_RATE"]

    # Initialize model and trainer
    autoencoder = AutoencoderFactory.get_autoencoder(model_type, params)
    trainer = AutoencoderTrainer(autoencoder, learning_rate=learning_rate)

    # Train model
    history, threshold = trainer.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        steps_per_epoch=steps_per_epoch,
        trial=trial
    )

    # Save training history
    os.makedirs(CONFIG["MODEL_PATH"], exist_ok=True)
    history_path = os.path.join(CONFIG["MODEL_PATH"], "history.pkl")
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    print(f"[INFO] Training history saved to {history_path}")

    # Save threshold
    threshold_path = os.path.join(CONFIG["MODEL_PATH"], "threshold.txt")
    with open(threshold_path, "w") as f:
        f.write(str(threshold))
    print(f"[INFO] Threshold fitted on validation set: {threshold:.6f}")

    return min(history.history["val_loss"])