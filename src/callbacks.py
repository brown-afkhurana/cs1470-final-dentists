import tensorflow as tf

from utils import generate_and_save_images, get_noise


class EpochVisualizer(tf.keras.callbacks.Callback):
    n_samples = 50
    def __init__(self, model, prefix='', **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.prefix = prefix
        self.sample_inputs = get_noise(model, self.n_samples)
        self.imgs = []

    def on_epoch_end(self, epoch, logs=None):
        generate_and_save_images(
            prefix=f'{self.prefix}{epoch}_', model=self.model, n_samples=self.n_samples, noise=self.sample_inputs)


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
    def __init__(self, model, patience=3, gen_increment=0.0002, metric='G_acc'):
        super().__init__()
        self.model = model
        self.patience = patience
        self.gen_increment = gen_increment
        self.metric = metric

        self.epoch_to_start_checking = -1
        self.hold_for_epochs = 5
        self.threshold = 0.01

        self.incremented = 0
        self.epochs_below_threshold = 0
        self.epochs_until_decrement = 1000

    def on_epoch_end(self, epoch, logs={}):
        if epoch < self.epoch_to_start_checking:
            return

        self.epochs_until_decrement -= 1

        if logs[self.metric] < self.threshold:
            self.epochs_below_threshold += 1
        else:
            self.epochs_below_threshold = 0
            # only decrease LR to normal if still behaving
            if self.epochs_until_decrement == 0 and self.incremented:
                # decrease generator LR
                match self.metric:
                    case 'G_acc':
                        old_value = self.model.generator_optimizer.learning_rate
                        new_value = self.model.generator_optimizer.learning_rate - self.incremented
                        print(f'\ndecreasing generator LR from {old_value} to {new_value}')
                        self.model.generator_optimizer.learning_rate.assign(new_value)
                        self.incremented = False
                    case 'D_acc_F':
                        old_value = self.model.discriminator_optimizer.learning_rate
                        new_value = self.model.discriminator_optimizer.learning_rate - self.incremented
                        print(f'\ndecreasing discriminator LR from {old_value} to {new_value}')
                        self.model.discriminator_optimizer.learning_rate.assign(new_value)
                        self.incremented = False
                    case _:
                        raise

        if self.epochs_below_threshold >= self.patience:
            # increase generator LR
            match self.metric:
                case 'G_acc':
                    old_value = self.model.generator_optimizer.learning_rate
                    new_value = self.model.generator_optimizer.learning_rate + self.gen_increment
                    print(f'\nincreasing generator LR from {old_value} to {new_value}')
                    self.model.generator_optimizer.learning_rate.assign(new_value)
                    self.incremented += self.gen_increment
                    self.epochs_below_threshold = 0
                    self.epochs_until_decrement = self.hold_for_epochs
                case 'D_acc_F':
                    old_value = self.model.discriminator_optimizer.learning_rate
                    new_value = self.model.discriminator_optimizer.learning_rate + self.gen_increment
                    print(f'\nincreasing discriminator LR from {old_value} to {new_value}')
                    self.model.discriminator_optimizer.learning_rate.assign(new_value)
                    self.incremented += self.gen_increment
                    self.epochs_below_threshold = 0
                    self.epochs_until_decrement = self.hold_for_epochs
                case _:
                    raise

class DiscriminatorSuspensionCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs['G_acc'] == 0:
            if self.model.discriminator.trainable:
                print(f'\nDiscriminator suspended at epoch {epoch}')
                self.model.discriminator.trainable = False
        else:
            if not self.model.discriminator.trainable:
                print(f'\nDiscriminator unsuspended at epoch {epoch}')
                self.model.discriminator.trainable = True
