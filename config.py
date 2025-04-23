CONFIG = {
    "DATA_PATH":"/home/cblasco/project_dat255_DL/DAT255Project_LHC_Anomaly_detection/data/events_anomalydetection.h5",
    #"DATA_PATH": "/Users/carlamiquelblasco/Desktop/MASTER SE/Q2/DAT255-DL/Project/DAT255Project_LHC_Anomaly_detection/data/events_anomalydetection.h5", #/Users/carlamiquelblasco/Desktop/MASTER BERGEN v2/Q2/DAT255-DL/Project/DAT255Project_LHC_Anomaly_detection/data/events_anomalydetection.h5",  # Path to dataset
    "MODEL_TYPE": "rnn",  # Change to "cnn" or "rnn" to try different models
    "MODE": "train",
    "PATIENCE": 4,
    "TEST_DATA_RATE": 0.02, # % of total data for test
    "TEST_ANOMALY_RATIO": 0.1,  # % anomalies in the test set
    "VALIDATION_DATA_RATE": 0.02,  # Rate of validation data within the train data
    "USE_OPTUNA": False, # Hyperparameter tuning with Optuna
    "OPTUNA_TRIALS": 1, # Number of Optuna trials (hyperparameter combinations)
    "EPOCHS":1,
    "BATCH_SIZE": 128,
    "LEARNING_RATE": 0.001, # Default learning rate
    "LEARNING_RATE_MIN": 0.00001,  # Minimum learning rate for Optuna
    "LEARNING_RATE_MAX": 0.01,  # Maximum learning rate for Optuna
    "CHECKPOINT_DIR": "./saved_models/checkpoints/",
    "FINAL_MODEL_DIR": "./saved_models/final/",
    "MODEL_PATH": "/Users/carlamiquelblasco/Desktop/MASTER SE/Q2/DAT255-DL/project_carla/VAE_Anomalie/saved_models/rnn/22042025",
    "THRESHOLD_PERCENTILE": 95,  # Used for anomaly detection classification. To classify as bckg or anomaly: threshold_percentile % of the events with less error are considered background
    "FEATURE_WEIGHTS":[3.0, 1.0, 1.0], # Feature-wise weights for computing the loss: [pT, eta, phi]

    #For plot test:
    "PLOTS": ['error_dist', 'event_comparison', 'latent', 'density', 'recon_vs_orig'],
    "LATENT_PLOT_MAX_POINTS":3000,
    "LATENT_PLOT_METHOD":"tsne", # or pca


    # Several options means Optuna will search for the best hyperparameters 
    # If not using Optuna, the first value in each list will be used
    "AUTOENCODER_PARAMS": { 
        "vae": {
            "input_shape": (700, 3),  # Sequential
            "encoder_layers": [512, 256, 128], # Encoder hidden layers
            "latent_dim": 16,  # Default dimensionality of latent space
            "latent_dim_min": 16, # Optuna
            "latent_dim_max": 128, # Optuna
            "latent_dim_step": 32, # Optuna
            "decoder_layers": [128, 256, 512], # Decoder hidden layers
            "activation": "relu",
            "beta": 0.0001,  # default weight for KL divergence loss
            "beta_min": 0.00001, # Optuna
            "beta_max": 0.001, # Optuna
            "dropout_rate": 0.1,
            "l2_strength": 1e-5
        },
        "cnn": {
            "input_shape":(700, 3, 1),
            "encoder_filters": [[64, 32, 16], [128, 64, 32]],
            "kernel_sizes": [[(5, 3), (3, 3), (3, 3)], [(3, 3), (3, 3), (3, 3)]],
            "latent_dim": 32,  # Default dimensionality of latent space
            "latent_dim_min": 32, # Optuna
            "latent_dim_max": 128, # Optuna
            "latent_dim_step": 32, # Optuna
            "activation_encoder": ['relu', 'tanh', 'sigmoid'],
            "activation_decoder": ['relu', 'tanh', 'sigmoid']
        },
        "rnn": {
            "input_shape":(700, 3),
            "rnn_type": ["GRU"],
            "encoder_layers":[[128, 64]],
            "latent_dim": 64, # Default dimensionality of latent space
            "latent_dim_min": 32, # Optuna
            "latent_dim_max": 128, # Optuna
            "latent_dim_step": 32, # Optuna
            "activation_encoder": ["sigmoid"], #['relu', 'tanh', 'sigmoid'],
            "activation_decoder": ["tanh"], #['relu', 'tanh', 'sigmoid']
        }
    }
}


