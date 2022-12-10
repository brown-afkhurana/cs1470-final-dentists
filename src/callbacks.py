import tensorflow as tf

from utils import generate_and_save_images, get_noise

class EpochVisualizer(tf.keras.callbacks.Callback):
    def __init__(self, model, prefix='', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.prefix = prefix
        self.sample_inputs = get_noise(model, 8)
        self.imgs = []

    def on_epoch_end(self, epoch, logs=None):
        generate_and_save_images(prefix=f'{self.prefix}{epoch}_', model=self.model, n_samples=8, noise=self.sample_inputs)

class StopDLossLow(tf.keras.callbacks.Callback):
    def __init__(self, threshold=0.01, patience=3):
        super().__init__()
        self.threshold = threshold
        self.patience = patience
        self.epochs_remaining = self.patience

    def on_epoch_end(self, epoch, logs={}):
        current = logs['D_loss']

        if current < self.threshold:
            self.epochs_remaining -= 1
        else:
            self.epochs_remaining = self.patience
        
        if self.epochs_remaining == 0:
            print(f"\nEpoch {epoch+1}: early stopping THR")
            self.model.stop_training = True

class LRUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, patience=3, gen_increment=0.0002):
        super().__init__()
        self.model = model
        self.patience = patience
        self.gen_increment = gen_increment

        self.incremented = 0

        self.epochs_below_threshold = 0
        self.epoch_to_start_checking = 5


    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.epoch_to_start_checking:
            return

        if logs['G_acc'] < 0.01:
            self.epochs_below_threshold += 1
        else:
            self.epochs_below_threshold = 0
        
        if self.epochs_below_threshold >= self.patience:
            # increase generator LR
            old_value = self.model.generator_optimizer.learning_rate
            new_value = self.model.generator_optimizer.learning_rate + self.gen_increment
            print(f'\nincreasing generator LR from {old_value} to {new_value}')
            self.model.generator_optimizer.learning_rate.assign(new_value)
            self.incremented += self.gen_increment  
        
        if epoch == 50 and self.incremented:
            # decrease generator LR
            old_value = self.model.generator_optimizer.learning_rate
            new_value = self.model.generator_optimizer.learning_rate - self.incremented
            print(f'\ndecreasing generator LR from {old_value} to {new_value}')
            self.model.generator_optimizer.learning_rate.assign(new_value)
            self.incremented = False
