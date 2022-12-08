import tensorflow as tf
import numpy as np

from preprocessing import get_data_CIFAR
from models import GAN, Generator, Discriminator


def main():
    train_cifar_images, train_cifar_labels = get_data_CIFAR('train', return_one_hot=False)
    zero_indices = tf.argmin(train_cifar_labels, axis=-1)
    train_cifar_images_0 = tf.gather(train_cifar_images, zero_indices)

    test_cifar_images, test_cifar_labels = get_data_CIFAR('test', return_one_hot=False)
    zero_indices = tf.argmin(test_cifar_labels, axis=-1)
    test_cifar_images_0 = tf.gather(test_cifar_images, zero_indices)

    generator = Generator(latent_shape=(100,),
                          generation_shape=(32, 32, 3))
    discriminator = Discriminator(generation_shape=(32, 32, 3),
                                  return_logits=True)
    gan = GAN(generator=generator,
              discriminator=discriminator)

    gan.compile()
    gan.fit(train_cifar_images_0,
            train_cifar_images_0,
            epochs=10,
            batch_size=128,
            # validation_data=test_cifar_images_0,
            )


if __name__ == '__main__':
    main()