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

Edit the `setting.conf` file to adjust model and training settings.

### 5. Run Training

```bash
python 
```

## Outputs

- Training & validation loss curves  
- Latent space visualizations  
- z-mean distribution  
- Reconstruction error density plots  
- Precision-Recall & ROC curves  
- Original vs. reconstructed event features (for anomalies and background)

All results are saved in the `results/` directory.
