import argparse

import tensorflow as tf
import numpy as np

from preprocessing import get_data_CIFAR
from models import GAN, Generator, Discriminator
from utils import generate_and_save_images


def load_gan_and_generate_images():
    # gan = tf.keras.models.load_model('models/gan')
    gan = GAN.load('models/gan')
    generate_and_save_images(gan)


def train_gan_and_generate_images():
    train_cifar_images, train_cifar_labels = get_data_CIFAR('train', return_one_hot=False)
    zero_indices = tf.argmin(train_cifar_labels, axis=-1)
    train_cifar_images_0 = tf.gather(train_cifar_images, zero_indices)

    test_cifar_images, test_cifar_labels = get_data_CIFAR('test', return_one_hot=False)
    zero_indices = tf.argmin(test_cifar_labels, axis=-1)
    test_cifar_images_0 = tf.gather(test_cifar_images, zero_indices)

    generator = Generator()
    discriminator = Discriminator()
    gan = GAN(generator=generator,
              discriminator=discriminator)
    gan.build(input_shape=(None, 100))
    gan.compile()
    history = gan.fit(train_cifar_images_0,
            train_cifar_images_0,
            epochs=2,
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
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    args = parser.parse_args()

    if args.train:
        train_gan_and_generate_images()
    else:
        load_gan_and_generate_images()


if __name__ == '__main__':
    main()