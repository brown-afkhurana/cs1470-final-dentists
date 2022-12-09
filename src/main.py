import argparse

import tensorflow as tf
import numpy as np

from preprocessing import get_data_CIFAR, get_data_MNIST
from models import GAN, Generator, Discriminator, CGAN
from utils import generate_and_save_images
from callbacks import EpochVisualizer, StopDLossLow


def load_gan_and_generate_images():
    # gan = tf.keras.models.load_model('models/gan')
    gan = GAN.load('models/gan')
    generate_and_save_images(gan)


def train_gan_and_generate_images(epochs=10):
    train_cifar_images, train_cifar_labels = get_data_CIFAR('train', return_one_hot=False)
    zero_indices = tf.where(train_cifar_labels == 0, axis=-1)
    train_cifar_images_0 = tf.gather(train_cifar_images, zero_indices)

    test_cifar_images, test_cifar_labels = get_data_CIFAR('test', return_one_hot=False)
    zero_indices = tf.where(test_cifar_labels == 0, axis=-1)
    test_cifar_images_0 = tf.gather(test_cifar_images, zero_indices)

    generator = Generator()
    discriminator = Discriminator(with_noise=False)
    gan = GAN(generator=generator,
              discriminator=discriminator)
    gan.build(input_shape=(None, 100))
    gan.compile()
    history = gan.fit(train_cifar_images_0,
            train_cifar_images_0,
            epochs=epochs,
            batch_size=128,
            # validation_data=test_cifar_images_0,
            )

    # this might fix saving issues?
    random_noise = gan.get_noise(1)
    gan(random_noise, training=False)

    # save the model to disk
    gan.save('models/gan')
    
    # generate and save images
    generate_and_save_images(gan)


def train_mnist(epochs=10,
                latent_size=100,
                gen_optimizer=tf.keras.optimizers.Adam(1e-3),
                disc_optimizer=tf.keras.optimizers.Adam(1e-3),
                batch_size=128,
                subset=None,
                callbacks: list=[],
                use_cgan=False):
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

    # because EpochVisualizer takes model as input
    if EpochVisualizer in callbacks:
        callbacks.remove(EpochVisualizer)
        callbacks = [EpochVisualizer(gan)] + callbacks

    history = gan.fit(train_mnist_images,
            train_mnist_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    return gan, history


def train_mnist_and_generate_images(epochs=10):
    train_mnist_images, train_mnist_labels = get_data_MNIST('train', return_one_hot=False)
    zero_indices = tf.where(train_mnist_labels == 0)
    train_mnist_images_0 = tf.gather(train_mnist_images, zero_indices)
    train_mnist_images_0 = tf.squeeze(train_mnist_images_0, axis=1)

    # save sample image
    tf.keras.preprocessing.image.save_img('images/sample.png', train_mnist_images_0[0], data_format='channels_last',
                                            file_format='png', scale=True)    

    test_mnist_images, test_mnist_labels = get_data_MNIST('test', return_one_hot=False)
    zero_indices = tf.argmin(test_mnist_labels, axis=-1)
    test_mnist_images_0 = tf.gather(test_mnist_images, zero_indices)

    Generator.generation_shape = (32, 32, 1)
    Discriminator.generation_shape = (32, 32, 1)
    generator = Generator()
    discriminator = Discriminator(with_noise=False)
    gan = GAN(generator=generator,
              discriminator=discriminator)
    gan.build(input_shape=(None, 100))
    gan.compile(tf.keras.optimizers.Adam(0.001, beta_1=0.9), 
                tf.keras.optimizers.Adam(0.001, beta_1=0.9))
    viz_callback = EpochVisualizer(gan)
    history = gan.fit(train_mnist_images_0,
            train_mnist_images_0,
            epochs=epochs,
            # epochs=1,
            batch_size=128,
            callbacks=[viz_callback],
            # validation_data=test_cifar_images_0,
            )

    # this might fix saving issues?
    random_noise = gan.get_noise(1)
    gan(random_noise, training=False)

    # save the model to disk
    gan.save('models/gan')
    
    # generate and save images
    generate_and_save_images(gan)

    return history


def tune_mnist_hyperparameters():
    gen_lr = 1e-3
    disc_lr = 1e-3
    d_loss_threshold = 0.01
    gen_increment = 1e-3

    for i in range(100):
        print(f'ITERATION {i}')
        print(f'{gen_lr=}')
        gen_optimizer = tf.keras.optimizers.Adam(gen_lr)
        disc_optimizer = tf.keras.optimizers.Adam(disc_lr)
        stop_early_callback = StopDLossLow(d_loss_threshold / 10)
        model, history = train_mnist(epochs=5,
                              gen_optimizer=gen_optimizer,
                              disc_optimizer=disc_optimizer,
                              subset=0,
                              callbacks=[stop_early_callback])
        D_loss = history.history['D_loss']
        if D_loss[-1] < d_loss_threshold:
            gen_lr += gen_increment
        else:
            model.save(f'models/gan_{i}')
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--tune', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--cgan', action='store_true')
    args = parser.parse_args()

    # if args.tune:
    #     tune_mnist_hyperparameters()
    # if args.train:
    #     # train_gan_and_generate_images(epochs=args.epochs)
    #     train_mnist_and_generate_images(epochs=args.epochs)
    # else:
    #     load_gan_and_generate_images()
    if args.train:
        if args.cgan:
            model, history = train_mnist(epochs=args.epochs,
                                   use_cgan=True)
        else:
            model, history = train_mnist(epochs=args.epochs,
                                   use_cgan=False,
                                   subset=0,
                                   callbacks=[EpochVisualizer],
                                   gen_optimizer=tf.keras.optimizers.Adam(0.0004, beta_1=0.5),
                                   disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.6))
    model.save('models/gan')


if __name__ == '__main__':
    main()