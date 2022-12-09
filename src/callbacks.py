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
    def __init__(self, threshold):
        super().__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs={}):
        current = logs['D_loss']

        if current < self.threshold:
            print(f"\nEpoch {epoch+1}: early stopping THR")
            self.model.stop_training = True

class LRUpdateCallback(tf.keras.callbacks.Callback):
    def __init__(self, model, gen_increment=0.0002):
        super().__init__()
        self.model = model
        self.gen_increment = gen_increment
        self.incremented = False

    def on_epoch_end(self, epoch, logs={}):
        if epoch == 15 and logs['D_loss'] < 0.001:
            # increase generator LR
            old_value = self.model.gen_optimizer.learning_rate
            new_value = self.model.gen_optimizer.learning_rate + self.gen_increment
            print(f'increasing generator LR from {old_value} to {new_value}')
            self.model.gen_optimizer.learning_rate.assign(new_value)
            self.incremented = True
        
        if epoch == 50 and self.incremented:
            # decrease generator LR
            old_value = self.model.gen_optimizer.learning_rate
            new_value = self.model.gen_optimizer.learning_rate - self.gen_increment
            print(f'decreasing generator LR from {old_value} to {new_value}')
            self.model.gen_optimizer.learning_rate.assign(new_value)
            self.incremented = False
