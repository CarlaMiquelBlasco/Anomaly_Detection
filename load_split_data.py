import h5py
import numpy as np
import hdf5plugin
import os
from sklearn.model_selection import train_test_split

# Define paths
DATASET_PATH = "/Users/carlamiquelblasco/Desktop/MASTER BERGEN v2/Q2/DAT255-DL/Project/data/events_anomalydetection.h5"
TRAIN_SAVE_PATH = "/Users/carlamiquelblasco/Desktop/MASTER BERGEN v2/Q2/DAT255-DL/Project/data/train/events_anomalydetection_train.h5"
TEST_SAVE_PATH = "/Users/carlamiquelblasco/Desktop/MASTER BERGEN v2/Q2/DAT255-DL/Project/data/test/events_anomalydetection_test.h5"

# Fraction of data to keep
SAMPLE_FRACTION = 0.3  # Keep 50% of the dataset


def load_hdf5_data_subset(file_path, sample_fraction=0.3):
    """ Loads only a subset of the dataset from HDF5 file to reduce memory usage. """
    print(f"[INFO] Loading dataset from: {file_path}")

    with h5py.File(file_path, "r") as f:
        total_samples = f["df"]["block0_values"].shape[0]  # Total number of rows
        sampled_indices = np.random.choice(total_samples, int(total_samples * sample_fraction), replace=False)
        sampled_indices.sort()  # Ensure indices are sorted to avoid HDF5 indexing errors

        data = f["df"]["block0_values"][sampled_indices]  # Load only selected rows

        labels = data[:, -1]  # Extract last column as labels (0=background, 1=signal)
        features = data[:, :-1]  # All columns except last are features

    print(f"[INFO] Loaded data shape: {features.shape}, Labels shape: {labels.shape}")
    return features, labels


def split_and_save_data(features, labels, train_path, test_path, test_size=0.2):
    """ Splits dataset into train and test sets, ensuring test contains all signals + some unseen background. """

    # 🔹 Separate background and signal events
    background_indices = np.where(labels == 0)[0]  # Indices where label = 0 (background)
    signal_indices = np.where(labels == 1)[0]  # Indices where label = 1 (signal)

    print(f"[INFO] Found {len(background_indices)} background events.")
    print(f"[INFO] Found {len(signal_indices)} signal events.")

    # 🔹 Extract background and signal data
    background_features = features[background_indices]
    signal_features = features[signal_indices]

    background_labels = labels[background_indices]
    signal_labels = labels[signal_indices]

    # 🔹 Split background into train (80%) and test (20%)
    train_background, test_background, train_labels, test_labels = train_test_split(
        background_features, background_labels, test_size=test_size, random_state=42
    )

    print(f"[INFO] Train Background: {train_background.shape}, Test Background: {test_background.shape}")

    # 🔹 Final Test Set = All Signals + Unseen Background
    test_features = np.vstack((signal_features, test_background))
    test_labels = np.hstack((signal_labels, test_labels))

    print(f"[INFO] Final Test Set: {test_features.shape} (Includes all signals + unseen background)")

    # 🔹 Save datasets
    save_hdf5(train_path, train_background, train_labels)
    save_hdf5(test_path, test_features, test_labels)

    print(f"[INFO] Datasets saved successfully:\n  → Train: {train_path}\n  → Test: {test_path}")


def save_hdf5(file_path, features, labels):
    """ Saves the dataset to an HDF5 file. """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)  # Ensure directory exists
    with h5py.File(file_path, "w") as f:
        data = np.column_stack((features, labels))  # Combine features & labels
        f.create_dataset("df/block0_values", data=data)
    print(f"[INFO] Saved dataset: {file_path}")


if __name__ == "__main__":
    features, labels = load_hdf5_data_subset(DATASET_PATH, SAMPLE_FRACTION)
    split_and_save_data(features, labels, TRAIN_SAVE_PATH, TEST_SAVE_PATH)
