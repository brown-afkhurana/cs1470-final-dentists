import argparse

import tensorflow as tf
import numpy as np

from preprocessing import get_data_CIFAR, get_data_MNIST, get_data_CIFAR10H, get_data_CIFAR10H_counts
from models import GAN, Generator, Discriminator, CGAN
from utils import generate_and_save_images
from callbacks import EpochVisualizer, StopDLossLow, LRUpdateCallback, DiscriminatorSuspensionCallback


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
    discriminator = Discriminator(with_noise=False, complex=True)

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
    if DiscriminatorSuspensionCallback in callbacks:
        callbacks.remove(DiscriminatorSuspensionCallback)
        callbacks = [DiscriminatorSuspensionCallback()] + callbacks

    history = gan.fit(train_mnist_images,
            train_mnist_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    if gan.stop_training:
        gan.stop_training = False
        gan.gen_steps = 2
        history = gan.fit(train_mnist_images,
                          train_mnist_labels,
                          epochs=epochs,
                          batch_size=batch_size,
                          callbacks=callbacks)
    if gan.stop_training:
        gan.stop_training = False
        gan.gen_steps = 3
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
    if LRUpdateCallback in callbacks:  # two LRupdate ==> for G and for D_F
        callbacks.remove(LRUpdateCallback)
        callbacks = [LRUpdateCallback(gan, patience=3, gen_increment=0.0002, metric='D_acc_F')] + callbacks
    if DiscriminatorSuspensionCallback in callbacks:
        callbacks.remove(DiscriminatorSuspensionCallback)
        callbacks = [DiscriminatorSuspensionCallback(gan)] + callbacks

    history = gan.fit(train_cifar_images,
            train_cifar_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    return gan, history


def train_cifar_10h(epochs=10,
                   latent_size=100,
                   gen_optimizer=tf.keras.optimizers.Adam(1e-3),
                   disc_optimizer=tf.keras.optimizers.Adam(1e-3),
                   batch_size=128,
                   subset=None,
                   callbacks: list=[],
                   viz_prefix=''):
    # load data
    train_cifar_images, train_cifar_labels = get_data_CIFAR10H('train', return_one_hot=False)
    # train_cifar_labels = tf.squeeze(train_cifar_labels, axis=1)
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
    gan = GAN(generator=generator,
            discriminator=discriminator)
    gan.build(input_shape=(None, latent_size))
    
    
    gan.compile(gen_optimizer, disc_optimizer)

    # instantiate callbacks that take model as input
    if EpochVisualizer in callbacks:
        callbacks.remove(EpochVisualizer)
        callbacks = [EpochVisualizer(gan, viz_prefix, save_every_n=20)] + callbacks

    history = gan.fit(train_cifar_images,
            train_cifar_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    return gan, history


def train_cifar_10h_counts(epochs=10,
                         latent_size=100,
                   gen_optimizer=tf.keras.optimizers.Adam(1e-3),
                   disc_optimizer=tf.keras.optimizers.Adam(1e-3),
                   batch_size=128,
                   subset=None,
                   callbacks: list=[],
                   viz_prefix=''):
    # load data
    train_cifar_images, train_cifar_labels = get_data_CIFAR10H_counts('train', return_one_hot=False)
    # train_cifar_labels = tf.squeeze(train_cifar_labels, axis=1)
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
    gan = GAN(generator=generator,
            discriminator=discriminator)
    gan.build(input_shape=(None, latent_size))
    
    
    gan.compile(gen_optimizer, disc_optimizer)

    # instantiate callbacks that take model as input
    if EpochVisualizer in callbacks:
        callbacks.remove(EpochVisualizer)
        callbacks = [EpochVisualizer(gan, viz_prefix, save_every_n=5)] + callbacks

    history = gan.fit(train_cifar_images,
            train_cifar_labels,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks)

    return gan, history

    
def train_all_mnist_gans(epochs=100):
    for i in range(10):
        # callbacks = [EpochVisualizer, LRUpdateCallback]
        # callbacks = [EpochVisualizer, DiscriminatorSuspensionCallback]
        callbacks = [EpochVisualizer, StopDLossLow(patience=10)]
        model, history = train_mnist(epochs=epochs,
                                     use_cgan=False,
                                     subset=i,
                                     callbacks=callbacks,
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0005, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     viz_prefix=f'mnist/{i}/')
        model.save(f'models/gan/mnist_{i}')


def train_mnist_gans_subset(subset: list[int], epochs=100):
    for i in subset:
        callbacks = [EpochVisualizer, StopDLossLow(patience=10)]
        model, history = train_mnist(epochs=epochs,
                                     use_cgan=False,
                                     subset=i,
                                     callbacks=callbacks,
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0005, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
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
                                        callbacks=[EpochVisualizer, StopDLossLow()],
                                        gen_optimizer=tf.keras.optimizers.Adam(0.0005, beta_1=0.5),
                                        disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.6),
                                        viz_prefix=f'mnist/{i}/')
            success = not model.stop_training
        if not success:
            print('Training failed. Moving on.')

        model.save(f'models/gan/mnist_{i}')


def retrain_mnist_gan_1(epochs=100):
    success = False
    tries = 0
    max_tries = 10
    while not success:
        tries += 1
        if tries > max_tries:
            break
        # load data
        train_mnist_images, train_mnist_labels = get_data_MNIST('train', return_one_hot=False)
        train_mnist_images = tf.gather(train_mnist_images, tf.where(train_mnist_labels == 1))
        train_mnist_labels = tf.gather(train_mnist_labels, tf.where(train_mnist_labels == 1))
        # axis fix
        train_mnist_images = tf.squeeze(train_mnist_images, axis=1)

        # model prep
        Generator.generation_shape = (32, 32, 1)
        Discriminator.generation_shape = (32, 32, 1)
        generator = Generator()
        discriminator = Discriminator(with_noise=False, complex=True)

        # build gan
        gan = GAN(generator=generator,
                discriminator=discriminator,
                gen_steps=3)
        gan.build(input_shape=(None, 100))
        
        
        gan.compile(tf.keras.optimizers.Adam(0.0005, beta_1=0.5),
                    tf.keras.optimizers.Adam(0.0002, beta_1=0.6),)

        # instantiate callbacks that take model as input
        callbacks = [EpochVisualizer(gan, 'mnist/1/'), StopDLossLow(threshold=0.005)]

        history = gan.fit(train_mnist_images,
                train_mnist_labels,
                epochs=epochs,
                batch_size=128,
                callbacks=callbacks)
        
        success = not gan.stop_training

    gan.save(f'models/gan/mnist_{1}')


def train_all_cifar_gans(epochs=100):
    for i in range(10):
        model, history = train_cifar(epochs=epochs,
                                     use_cgan=False,
                                     subset=i,
                                    #  callbacks=[EpochVisualizer, LRUpdateCallback, LRUpdateCallback],
                                     callbacks=[EpochVisualizer],
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     viz_prefix=f'cifar/{i}/')
        model.save(f'models/gan/cifar_{i}')


def train_all_cifar_10h_gans(epochs=100):
    for i in range(10):
        model, history = train_cifar_10h(epochs=epochs,
                                     subset=i,
                                     callbacks=[EpochVisualizer],
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     viz_prefix=f'cifar10h/{i}/')
        model.save(f'models/gan/cifar10h_{i}')

def train_all_cifar_10h_gans_counts(epochs=100):
    for i in range(10):
        model, history = train_cifar_10h_counts(epochs=epochs,
                                     subset=i,
                                     callbacks=[EpochVisualizer],
                                     gen_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     disc_optimizer=tf.keras.optimizers.Adam(0.0002, beta_1=0.5),
                                     viz_prefix=f'cifar10h_counts/{i}/')
        model.save(f'models/gan/cifar10h_counts_{i}')


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--train', action='store_true')
    # parser.add_argument('--tune', action='store_true')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='cifar')
    parser.add_argument('--subset', type=int, default=None, nargs='+')
    # parser.add_argument('--cgan', action='store_true')
    args = parser.parse_args()

    match args.dataset, args.subset:
        case 'mnist', None:
            train_all_mnist_gans(epochs=args.epochs)
        case 'cifar', None:
            train_all_cifar_gans(epochs=args.epochs)
        case 'cifar10h', None:
            train_all_cifar_10h_gans(epochs=args.epochs)
        case 'cifar10h_counts', None:
            train_all_cifar_10h_gans_counts(epochs=args.epochs)
        case 'mnist', [1]:
            retrain_mnist_gan_1(epochs=args.epochs)
        case 'mnist', list():
            # retrain_mnist_gans_subset(args.subset, epochs=args.epochs)
            train_mnist_gans_subset(args.subset, epochs=args.epochs)
        case 'cifar', list():
            raise NotImplementedError('cifar retrain not implemented yet')

# python src\train.py --dataset cifar10h_counts --epochs 40


if __name__ == '__main__':
    main()