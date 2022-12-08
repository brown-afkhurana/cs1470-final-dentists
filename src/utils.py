import os

import tensorflow as tf
import imageio

from models import GAN, Generator, Discriminator

SEED = 26498

def get_noise(model: GAN, n_samples):
    # return model.noise_generator.normal(shape=(n_samples, *model.generator.hardcode_input_shape))
    noise_generator = tf.random.Generator.from_seed(SEED)
    return noise_generator.normal(shape=(n_samples, *Generator.latent_shape))

def generate_and_save_images(model: GAN, n_samples=5):
    # generate 5 samples of the latent space
    latent_samples = get_noise(model, n_samples)

    # generate the images
    generated_images = model.generator(latent_samples)

    # create directory if not exists
    if not os.path.exists('images'):
        os.makedirs('images')

    # save the images to disk
    image_filenames = []
    for i in range(n_samples):
        image = generated_images[i]
        image = tf.cast(image, tf.uint8)
        image_filename = f'images/{i}.png'
        tf.keras.preprocessing.image.save_img(image_filename, image)
