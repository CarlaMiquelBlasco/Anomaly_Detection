import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import argparse
import numpy as np
import optuna
import hdf5plugin  # Required to read compressed HDF5 files

from config import CONFIG
from models.autoencoders_factory import AutoencoderFactory
from data_processing.data_loader import DataLoader
from models.train import AutoencoderTrainer
from models.test import AutoencoderTester
from utils.visualization import log_optuna_study, plot_results, analyze_distribution
from hyperparameter_tuning.hyperparameter_tuning import objective
from utils.file_utils import get_model_filename, get_latest_model
from utils.run_plots import run_all_plots


def parse_arguments():
    """
    Parses command-line arguments for configuring the anomaly detection pipeline.
    """
    parser = argparse.ArgumentParser(description="LHC Anomaly Detection Pipeline")

    parser.add_argument("--mode", type=str, choices=["train", "test"], default=CONFIG["MODE"],
                        help="Operation mode: 'train' to train the model or 'test' to evaluate a trained model.")

    parser.add_argument("--data_path", type=str, default=CONFIG["DATA_PATH"],
                        help="Path to the dataset (HDF5 file)")

    parser.add_argument("--model_type", type=str, choices=["mlp", "cnn", "rnn"], default=CONFIG["MODEL_TYPE"],
                        help="Type of autoencoder model to use")

    parser.add_argument("--epochs", type=int, default=CONFIG["EPOCHS"],
                        help="Number of training epochs")

    parser.add_argument("--batch_size", type=int, default=CONFIG["BATCH_SIZE"],
                        help="Batch size for training")

    parser.add_argument("--threshold_percentile", type=int, default=CONFIG["THRESHOLD_PERCENTILE"],
                        help="Percentile used to determine the anomaly threshold")

    parser.add_argument("--model_path", type=str, default=CONFIG["MODEL_PATH"],
                        help="Path to a trained model (.keras). If not specified, the latest model is used.")

    parser.add_argument("--use_optuna", type=bool, default=CONFIG["USE_OPTUNA"],
                        help="Enable Optuna for hyperparameter tuning")

    parser.add_argument("--optuna_trials", type=int, default=CONFIG["OPTUNA_TRIALS"],
                        help="Number of Optuna trials to perform")

    parser.add_argument("--plots", nargs="+", default=CONFIG["PLOTS"],
                        help="List of plots to generate. Options: 'error_dist', 'event_comparison', 'latent', 'density', 'recon_vs_orig'")

    return parser.parse_args()


def main():
    """
    Main entry point for training or testing the autoencoder pipeline.
    """
    args = parse_arguments()

    # Update configuration using provided arguments
    CONFIG["DATA_PATH"] = args.data_path
    CONFIG["MODEL_TYPE"] = args.model_type
    CONFIG["EPOCHS"] = args.epochs
    CONFIG["BATCH_SIZE"] = args.batch_size
    CONFIG["THRESHOLD_PERCENTILE"] = args.threshold_percentile
    CONFIG["USE_OPTUNA"] = args.use_optuna
    CONFIG["OPTUNA_TRIALS"] = args.optuna_trials

    print("\n<<< LHC Anomaly Detection Pipeline >>>")
    print(f"Mode: {args.mode}")
    if args.mode == "train":
        print(f"Dataset: {CONFIG['DATA_PATH']}")
        print(f"Model type: {CONFIG['MODEL_TYPE']}")
        print(f"Epochs: {CONFIG['EPOCHS']}, Batch size: {CONFIG['BATCH_SIZE']}")
        print(f"Anomaly threshold percentile: {CONFIG['THRESHOLD_PERCENTILE']}")
    print("--------------------------------------")

    if args.mode == "train":
        CONFIG["MODEL_PATH"] =  f"./saved_models/{get_model_filename()}_dir"
        os.makedirs(CONFIG["MODEL_PATH"], exist_ok=True)

        print(f"Training mode enabled. Models will be saved to: {CONFIG['MODEL_PATH']}")

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=args.optuna_trials)

        print("\nTraining completed.")
        print(f"Best hyperparameters: {study.best_trial.params}")
        print(f"Validation loss: {study.best_trial.value:.6f}")

        log_optuna_study(study)

    elif args.mode == "test":
        if args.model_path is not None:
            CONFIG["MODEL_PATH"] = args.model_path
            print(f"[INFO] Loading specified model: {CONFIG['MODEL_PATH']}")
        else:
            CONFIG["MODEL_PATH"] = get_latest_model()
            print(f"[INFO] No model path provided. Using latest model: {CONFIG['MODEL_PATH']}")

        # Load test data
        data_loader = DataLoader(CONFIG["MODEL_PATH"])
        data_loader.prepare_datasets()
        test_dataset = data_loader.get_test_dataset()

        # Initialize and load the model
        model_params = CONFIG["AUTOENCODER_PARAMS"][args.model_type]
        autoencoder = AutoencoderFactory.get_autoencoder(args.model_type, model_params)

        print("\nModel summary:")
        autoencoder.summary()

        # Run evaluation
        tester = AutoencoderTester(autoencoder, CONFIG["MODEL_PATH"])
        errors_test, threshold_test, predictions_test, history, labels = tester.test(test_dataset)

        # Plot results
        model = tester.get_model()
        run_all_plots(errors_test, threshold_test, labels, history, args.plots, test_dataset, model)


if __name__ == "__main__":
    main()
