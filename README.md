# Anomaly Detection in Particle Physics with VAEs

This project explores the use of **Variational Autoencoders (VAEs)** to detect anomalous events in high-energy particle collision data. The goal is to train a model that accurately reconstructs background (normal) events while giving poor reconstructions for anomalies — allowing us to identify them based on high reconstruction error.

We use a deep learning pipeline built with **TensorFlow/Keras** and perform hyperparameter tuning using **Optuna**. The dataset consists of sequences of particle features: **transverse momentum (pT)**, **pseudorapidity (η)**, and **azimuthal angle (φ)**.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set Up a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 3. Install Required Packages

```bash
pip install -r requirements.txt
```

### 4. Configure the Run

Edit the `config.py` file to adjust data, model and training settings to set parameters such as:

```python
DATA_PATH = "your_dataset_path"
MODEL_TYPE = "vae"         # "vae", "rnn", or "cnn"
MODE = "train"             # "train" or "test"
MODEL_PATH = "path/to/model_dir"  # Required for MODE="test"
```

You can also configure:
- Encoder/decoder layers
- Latent dimension
- Dropout, beta (KL weight), and regularization
- Learning rate and optimizer settings
- Evaluation plots


### 5. Run Training

To train a model (e.g. VAE):

```bash
python main.py --mode train --model_type vae
```

This will:
- Load and preprocess the data
- Train the model using background-only events
- Save the model and scaler to `./saved_models/`
- (Optional) Run Optuna for hyperparameter tuning if enabled

### Evaluate a Model

To evaluate a **pretrained model**, set:

```python
MODE = "test"
MODEL_TYPE = "vae" | "rnn" | "cnn"
MODEL_PATH = "./saved_models/your_model"
```

Then run:

```bash
python main.py --mode test --model_type vae
```

This will:
- Load the model and scalers
- Compute reconstruction errors on test data
- Classify events using a threshold (95th percentile on validation error)
- Output accuracy, precision, recall, F1
- Generate diagnostic plots


## Plot Options

You can choose which evaluation plots to generate by setting the `PLOTS` list in `config.py`. Default:

```python
PLOTS = ['error_dist', 'event_comparison', 'latent', 'density', 'recon_vs_orig']
```

Available options:
- `error_dist`: Histogram + ROC + Precision-Recall curve
- `event_comparison`: Compare original anomalous and normal events
- `latent`: 2D latent space visualization using PCA or t-SNE
- `density`: KDE plots of errors and latent space
- `recon_vs_orig`: Feature-wise plots of reconstructed vs. original particles

---

## Output

Saved model assets include:
- `model.keras`: Trained Keras model
- `scalers.pkl`: Feature normalizers
- `threshold.txt`: Error threshold used for classification
- `history.pkl`: Training/validation loss history

## Example Commands

Train a VAE and plot error distribution:
```bash
python main.py --mode train --model_type vae
```

Evaluate a trained RNN and visualize latent space:
```bash
python main.py --mode test --model_type rnn --model_path ./saved_models/rnn_model --plots latent
```


