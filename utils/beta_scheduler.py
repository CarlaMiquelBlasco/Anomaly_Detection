import tensorflow as tf

class BetaWarmupScheduler(tf.keras.callbacks.Callback):
    def __init__(self, model_reference, max_beta=1.0, warmup_epochs=15):
        super().__init__()
        self._model_ref = model_reference
        self.max_beta = max_beta
        self.warmup_epochs = warmup_epochs

    def on_epoch_begin(self, epoch, logs=None):
        new_beta = min(self.max_beta, self.max_beta * (epoch + 1) / self.warmup_epochs)
        self.model.beta = new_beta
        print(f"[BETA WARMUP] Epoch {epoch+1}: Setting beta to {self.model.beta:.5f}")
