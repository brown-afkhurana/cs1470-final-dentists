import argparse

import tensorflow as tf
import numpy as np

from preprocessing import get_data_CIFAR, get_data_MNIST
from models import GAN, Generator, Discriminator, CGAN
from utils import generate_and_save_images
from callbacks import EpochVisualizer, StopDLossLow, LRUpdateCallback


def train_mnist(epochs=10,
                latent_size=100,
                gen_optimizer=tf.keras.optimizers.Adam(1e-3),
                disc_optimizer=tf.keras.optimizers.Adam(1e-3),
                batch_size=128,
                subset=None,
                callbacks: list=[],
                use_cgan=False,
                viz_prefix=''):
    # load data
    train_mnist_images, train_mnist_labels = get_data_MNIST('train', return_one_hot=False)
    if subset is not None:
        train_mnist_images = tf.gather(train_mnist_images, tf.where(train_mnist_labels == subset))
        train_mnist_labels = tf.gather(train_mnist_labels, tf.where(train_mnist_labels == subset))
        # axis fix
        train_mnist_images = tf.squeeze(train_mnist_images, axis=1)

    # model prep
    Generator.generation_shape = (32, 32, 1)
    Discriminator.generation_shape = (32, 32, 1)
    generator = Generator()
    discriminator = Discriminator(with_noise=False)

    # build gan
    if use_cgan:
        gan = CGAN(generator, discriminator)
        gan.build(input_shape=(None, 2 * latent_size))
    else:
        gan = GAN(generator=generator,
                discriminator=discriminator)
        gan.build(input_shape=(None, latent_size))
    
    
    gan.compile(gen_optimizer, disc_optimizer)

    # instantiate callbacks that take model as input
    if EpochVisualizer in callbacks:
        callbacks.remove(EpochVisualizer)
        callbacks = [EpochVisualizer(gan, viz_prefix)] + callbacks
    if LRUpdateCallback in callbacks:
        callbacks.remove(LRUpdateCallback)
        callbacks = [LRUpdateCallback(gan, patience=3, gen_increment=0.0002)] + callbacks

    history = gan.fit(train_mnist_images,
            train_mnist_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    return gan, history


def train_cifar(epochs=10,
                latent_size=100,
                gen_optimizer=tf.keras.optimizers.Adam(1e-3),
                disc_optimizer=tf.keras.optimizers.Adam(1e-3),
                batch_size=128,
                subset=None,
                callbacks: list=[],
                use_cgan=False,
                viz_prefix=''):
    # load data
    train_cifar_images, train_cifar_labels = get_data_CIFAR('train', return_one_hot=False)
    train_cifar_labels = tf.squeeze(train_cifar_labels, axis=1)
    if subset is not None:
        train_cifar_images = tf.gather(train_cifar_images, tf.where(train_cifar_labels == subset))
        train_cifar_labels = tf.gather(train_cifar_labels, tf.where(train_cifar_labels == subset))
        # axis fix
        train_cifar_images = tf.squeeze(train_cifar_images, axis=1)
    
    # model prep
    Generator.generation_shape = (32, 32, 3)
    Discriminator.generation_shape = (32, 32, 3)
    generator = Generator()
    discriminator = Discriminator(with_noise=False, dropout=0.2, complex=True)

    # build gan
    if use_cgan:
        gan = CGAN(generator, discriminator)
        gan.build(input_shape=(None, 2 * latent_size))
    else:
        gan = GAN(generator=generator,
                discriminator=discriminator)
        gan.build(input_shape=(None, latent_size))
    
    
    gan.compile(gen_optimizer, disc_optimizer)

    # instantiate callbacks that take model as input
    if EpochVisualizer in callbacks:
        callbacks.remove(EpochVisualizer)
        callbacks = [EpochVisualizer(gan, viz_prefix)] + callbacks
    if LRUpdateCallback in callbacks:
        callbacks.remove(LRUpdateCallback)
        callbacks = [LRUpdateCallback(gan, patience=3, gen_increment=0.0002, metric='G_acc')] + callbacks
    if LRUpdateCallback in callbacks:  # still
        callbacks.remove(LRUpdateCallback)
        callbacks = [LRUpdateCallback(gan, patience=3, gen_increment=0.0002, metric='D_acc_F')] + callbacks

    history = gan.fit(train_cifar_images,
            train_cifar_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    return gan, history

    
def train_all_mnist_gans(epochs=100):
    for i in range(10):
        model, history = train_mnist(epochs=epochs,
                                     use_cgan=False,
                                     subset=i,
                                     callbacks=[EpochVisualizer, LRUpdateCallback],
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0005, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.6),
                                     viz_prefix=f'mnist/{i}/')
        model.save(f'models/gan/mnist_{i}')

def retrain_mnist_gans_subset(subset: list[int], epochs=100):
    for i in subset:
        success = False
        tries = 0
        max_tries = 10
        while not success:
            tries += 1
            if tries > max_tries:
                break
            model, history = train_mnist(epochs=epochs,
                                        use_cgan=False,
                                        subset=i,
                                        callbacks=[EpochVisualizer, LRUpdateCallback],
                                        gen_optimizer=tf.keras.optimizers.Adam(0.0005, beta_1=0.5),
                                        disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.6),
                                        viz_prefix=f'mnist/{i}/')
            success = not model.stop_training
        if not success:
            print('Training failed. Moving on.')

        model.save(f'models/gan/mnist_{i}')

def train_all_cifar_gans(epochs=100):
    for i in range(10):
        model, history = train_cifar(epochs=epochs,
                                     use_cgan=False,
                                     subset=i,
                                     callbacks=[EpochVisualizer, LRUpdateCallback, LRUpdateCallback],
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     viz_prefix=f'cifar/{i}/')
        model.save(f'models/gan/cifar_{i}')


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--tune', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    # parser.add_argument('--cgan', action='store_true')
    args = parser.parse_args()

    train_all_cifar_gans(epochs=args.epochs)


if __name__ == '__main__':
    main()