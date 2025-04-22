import os
import pickle
import h5py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from config import CONFIG
from tqdm import tqdm
import matplotlib.pyplot as plt


class DataLoader:
    def __init__(self, model_dir):
        self.data_path = CONFIG["DATA_PATH"]
        self.scalers = [StandardScaler()] + [MinMaxScaler() for _ in range(2)]  # For pT, η, φ
        self.model_dir = model_dir

        self.test_indices = []
        self.test_labels = []
        self.val_indices = []
        self.train_indices = []

    def save_scalers(self, path=None):
        path = path or self.model_dir
        with open(os.path.join(path, "scalers.pkl"), "wb") as f:
            pickle.dump(self.scalers, f)

    def load_scalers(self, path=None):
        path = path or self.model_dir
        with open(os.path.join(path, "scalers.pkl"), "rb") as f:
            self.scalers = pickle.load(f)

    def _apply_scalers(self, arr):
        # Detect zero-padded rows
        real_mask = ~np.all(arr == 0, axis=1)

        for i in range(3):
            feature = arr[:, i]
            feature[real_mask] = self.scalers[i].transform(feature[real_mask].reshape(-1, 1)).flatten()
            feature[~real_mask] = 0.0  # Reset padded rows

        return arr

    def prepare_datasets(self):
        print("[INFO] Preparing dataset and fitting scalers...")
        test_rate = CONFIG["TEST_DATA_RATE"]
        val_rate = CONFIG["VALIDATION_DATA_RATE"]
        anomaly_ratio = CONFIG["TEST_ANOMALY_RATIO"]

        signal_indices = []
        background_indices = []

        with h5py.File(self.data_path, "r") as f:
            data = f["df"]["block0_values"]
            for i in tqdm(range(data.shape[0])):
                label = data[i, -1]
                if label == 1:
                    signal_indices.append(i)
                else:
                    background_indices.append(i)

        np.random.seed(43)
        np.random.shuffle(signal_indices)
        np.random.shuffle(background_indices)

        total_samples = len(signal_indices) + len(background_indices)
        num_signal_test = int(anomaly_ratio * test_rate * total_samples)
        num_background_test = int((1 - anomaly_ratio) * test_rate * total_samples)
        val_size = int(val_rate * total_samples)

        # BUILD TEST DATASET
        self.test_indices = signal_indices[:num_signal_test] + background_indices[:num_background_test]
        self.test_labels = [1] * num_signal_test + [0] * num_background_test
        combined = list(zip(self.test_indices, self.test_labels))
        np.random.shuffle(combined)
        self.test_indices, self.test_labels = zip(*combined)

        #BUILD TRAIN AND VALIDATION (ONLY BCKG)
        background_remaining = background_indices[num_background_test:]
        self.val_indices = background_remaining[:val_size]
        self.train_indices = background_remaining[val_size:]

        if CONFIG["MODE"] == "trainn":
            all_particles = []

            with h5py.File(self.data_path, "r") as f:
                data = f["df"]["block0_values"]
                for idx in tqdm(self.train_indices):
                    row = data[idx][:-1].reshape(700, 3)

                    # Filter out zero-padded particles for fitting the scaler
                    non_zero_particles = row[~np.all(row == 0, axis=1)]
                    all_particles.append(non_zero_particles)

            all_particles = np.concatenate(all_particles, axis=0)  # Shape: (N, 3)
            print(f"[DEBUG] Fitting scalers on {all_particles.shape[0]} real particles (non-padded)")

            for i in range(3):
                self.scalers[i].fit(all_particles[:, i].reshape(-1, 1))

            self.save_scalers(CONFIG["MODEL_PATH"])
        else:
            #model aux only used when we wanted to train using fitted scalers from previous trials so we don't have to wait
            #model_aux = "/Users/carlamiquelblasco/Desktop/MASTER SE/Q2/DAT255-DL/project_carla/VAE_Anomalie/saved_models/vae/13042025"
            #self.load_scalers(model_aux)
            self.load_scalers(CONFIG["MODEL_PATH"])

        print("\n[INFO] Fitted scaler stats:")
        for i, name in enumerate(['pT', 'η', 'φ']):
            scaler = self.scalers[i]
            if isinstance(scaler, MinMaxScaler):
                print(f"  {name}: min={scaler.data_min_[0]:.4f}, max={scaler.data_max_[0]:.4f}, "
                      f"range={scaler.data_range_[0]:.4f}")
            elif isinstance(scaler, StandardScaler):
                print(f"  {name}: mean={scaler.mean_[0]:.4f}, std={scaler.scale_[0]:.4f}")
            else:
                print(f"  {name}: Unknown scaler type")

    def _transform_event_with_mask(self, row_flat):
        row = row_flat.reshape(700, 3)
        mask = (np.any(row != 0, axis=-1)).astype(np.float32).reshape(700, 1) # Generates a truth mask where active particles are 1.0, and zero-padded rows are 0.0
        scaled = self._apply_scalers(row.copy()) # Applies scaling per feature (pT, η, φ).
        return scaled.astype(np.float32), mask

    def get_lazy_train_dataset(self):
        def generator():
            with h5py.File(self.data_path, "r") as f:
                data = f["df"]["block0_values"]
                for idx in self.train_indices:
                    row = data[idx][:-1]
                    scaled, mask = self._transform_event_with_mask(row)
                    yield scaled, mask

        output_signature = (
            tf.TensorSpec(shape=(700, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(700, 1), dtype=tf.float32)
        )
        dataset = tf.data.Dataset.from_generator(generator, output_signature=output_signature)

        return dataset.batch(CONFIG["BATCH_SIZE"]).repeat().prefetch(tf.data.AUTOTUNE)

    def get_validation_dataset(self):
        val_features = []
        val_masks = []
        with h5py.File(self.data_path, "r") as f:
            data = f["df"]["block0_values"]
            for idx in self.val_indices:
                row = data[idx][:-1]
                scaled, mask = self._transform_event_with_mask(row)
                val_features.append(scaled)
                val_masks.append(mask)

        val_features = np.array(val_features, dtype=np.float32)
        val_masks = np.array(val_masks, dtype=np.float32)

        return tf.data.Dataset.from_tensor_slices((val_features, val_masks)).batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)

    def get_test_dataset(self):
        test_features = []
        test_masks = []
        with h5py.File(self.data_path, "r") as f:
            data = f["df"]["block0_values"]
            for idx in self.test_indices:
                row = data[idx][:-1]
                scaled, mask = self._transform_event_with_mask(row)
                test_features.append(scaled)
                test_masks.append(mask)

        test_features = np.array(test_features, dtype=np.float32)
        test_masks = np.array(test_masks, dtype=np.float32)


        return tf.data.Dataset.from_tensor_slices(((test_features, test_masks), np.array(self.test_labels))).batch(CONFIG["BATCH_SIZE"]).prefetch(tf.data.AUTOTUNE)
