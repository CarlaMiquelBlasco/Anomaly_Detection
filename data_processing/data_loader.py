import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

from config import CONFIG


class DataLoader:
    """
    Handles loading and preprocessing of LHC dataset for training, validation, and testing.
    Supports chunked loading for memory-efficient training.
    """

    def __init__(self, model_dir, num_samples=None):
        self.chunk_size = num_samples if num_samples else CONFIG["CHUNK_SIZE"]
        self.validation_data_rate = CONFIG["VALIDATION_DATA_RATE"]
        self.test_data_rate = CONFIG["TEST_DATA_RATE"]
        self.validation_anomaly_ratio = CONFIG["VALIDATION_ANOMALY_RATIO"]
        self.test_anomaly_ratio = CONFIG["TEST_ANOMALY_RATIO"]
        self.data_path = CONFIG["DATA_PATH"]
        self.scaler = MinMaxScaler()
        self.model_dir = model_dir
        self.background_offset = None  # Will be set after loading validation/test

    def save_scaler(self, path=None):
        path = path or self.model_dir
        with open(os.path.join(path, "scaler.pkl"), "wb") as f:
            pickle.dump(self.scaler, f)
        print(f"[INFO] Scaler saved to {os.path.join(path, 'scaler.pkl')}")

    def load_scaler(self, path=None):
        path = path or self.model_dir
        with open(os.path.join(path, "scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        print(f"[INFO] Scaler loaded from {os.path.join(path, 'scaler.pkl')}")

    def load_hdf5_data(self, chunk_size=1000, max_samples=None, start_offset=0):
        with h5py.File(self.data_path, "r") as f:
            data = f["df"]["block0_values"]
            for i in range(start_offset, data.shape[0], chunk_size):
                chunk = data[i:i + chunk_size]
                yield chunk
                if max_samples and i + chunk_size - start_offset >= max_samples:
                    break

    def to_tf_dataset(self, data, labels=None, shuffle=True):
        batch_size = CONFIG["BATCH_SIZE"]
        dataset = tf.data.Dataset.from_tensor_slices((data, labels) if labels is not None else (data, data))
        if shuffle:
            dataset = dataset.shuffle(buffer_size=len(data))
        return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    def preprocess_data(self, mode, offset):
        """
        Loads a chunk of background training data starting from the given offset.
        Only 'train' mode is supported in chunked mode.
        """
        if mode != "train":
            raise ValueError("Only 'train' mode is supported in chunked training.")

        background_data = []
        accumulated = 0

        for chunk in self.load_hdf5_data(chunk_size=10000, start_offset=offset):
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)

            background_chunk = chunk[chunk[:, -1] == 0]
            background_data.append(background_chunk)
            accumulated += background_chunk.shape[0]

            if accumulated >= self.chunk_size:
                break

        background_data = np.vstack(background_data)[:self.chunk_size]
        np.random.shuffle(background_data)

        train_features = background_data[:, :-1]
        self.scaler.fit(train_features)
        train_features = self.scaler.transform(train_features).reshape(-1, 700, 3)

        print("[DEBUG] Training data shape:", train_features.shape)
        print("[DEBUG] Training feature min/max:", np.min(train_features), np.max(train_features))
        print(f"[DEBUG] Loaded chunk of shape: {background_data.shape}")
        print(f"[DEBUG] Scaled data min: {train_features.min():.4f}, max: {train_features.max():.4f}")

        return self.to_tf_dataset(train_features)

    def load_fixed_validation_and_test_sets(self):
        """
        Loads fixed-size validation and test sets containing background and signal samples.
        Returns the datasets along with background offset for training.
        """
        print("[INFO] Loading validation and test sets...")

        signal_data = []
        background_data = []

        max_total = CONFIG.get("MAX_VAL_TEST_SAMPLES", 250000)
        total_val = int(self.validation_data_rate * max_total)
        total_test = int(self.test_data_rate * max_total)

        num_signal_val = int(self.validation_anomaly_ratio * total_val)
        num_background_val = total_val - num_signal_val
        num_signal_test = int(self.test_anomaly_ratio * total_test)
        num_background_test = total_test - num_signal_test

        needed_signal = num_signal_val + num_signal_test
        needed_background = num_background_val + num_background_test

        loaded_signal = loaded_background = rows_read = 0

        for chunk in self.load_hdf5_data(chunk_size=10000, start_offset=0):
            if chunk.ndim == 1:
                chunk = chunk.reshape(1, -1)

            sig = chunk[chunk[:, -1] == 1]
            bg = chunk[chunk[:, -1] == 0]

            if loaded_signal < needed_signal:
                signal_data.append(sig)
                loaded_signal += sig.shape[0]

            if loaded_background < needed_background:
                background_data.append(bg)
                loaded_background += bg.shape[0]
                rows_read += chunk.shape[0]

            if loaded_signal >= needed_signal and loaded_background >= needed_background:
                break

        self.background_offset = rows_read

        signal_data = np.vstack(signal_data)[:needed_signal]
        background_data = np.vstack(background_data)[:needed_background]
        np.random.shuffle(signal_data)
        np.random.shuffle(background_data)

        # Split into validation/test sets
        val_signal = signal_data[:num_signal_val]
        val_background = background_data[:num_background_val]
        test_signal = signal_data[num_signal_val:num_signal_val + num_signal_test]
        test_background = background_data[num_background_val:num_background_val + num_background_test]

        val_data = np.vstack((val_signal, val_background))
        test_data = np.vstack((test_signal, test_background))

        np.random.shuffle(val_data)
        np.random.shuffle(test_data)

        val_features = val_data[:, :-1]
        val_labels = val_data[:, -1]
        test_features = test_data[:, :-1]
        test_labels = test_data[:, -1]

        # Fit and apply scaler
        self.scaler.fit(background_data[:, :-1])
        self.save_scaler(CONFIG["MODEL_PATH"])
        val_features = self.scaler.transform(val_features).reshape(-1, 700, 3)
        test_features = self.scaler.transform(test_features).reshape(-1, 700, 3)

        # Return datasets
        print(f"[INFO] Validation set: {len(val_signal)} signal, {len(val_background)} background")
        print(f"[INFO] Test set: {len(test_signal)} signal, {len(test_background)} background")

        val_dataset_back = self.to_tf_dataset(val_features[val_labels == 0], shuffle=False)
        val_dataset_back_labeled = self.to_tf_dataset(val_features[val_labels == 0], labels=val_labels[val_labels == 0], shuffle=False)
        val_dataset_full = self.to_tf_dataset(val_features, labels=val_labels, shuffle=False)
        test_dataset = self.to_tf_dataset(test_features, labels=test_labels, shuffle=False)

        print("[DEBUG] Example scaled features (val):", val_features[0][:5])
        print("[DEBUG] Label distribution in validation:", np.unique(val_labels, return_counts=True))
        print("[DEBUG] Label distribution in test:", np.unique(test_labels, return_counts=True))

        return (
            val_dataset_back,
            val_dataset_back_labeled,
            val_dataset_full,
            val_labels,
            test_dataset,
            test_labels,
            self.background_offset
        )
