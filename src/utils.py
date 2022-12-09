import os

import tensorflow as tf

from models import GAN, Generator, Discriminator

SEED = 26498

def get_noise(model: GAN, n_samples):
    # return model.noise_generator.normal(shape=(n_samples, *model.generator.hardcode_input_shape))
    noise_generator = tf.random.Generator.from_seed(SEED)
    return noise_generator.normal(shape=(n_samples, *Generator.latent_shape))

def generate_and_save_images(model: GAN, n_samples=5, prefix='', noise=None):
    # generate 5 samples of the latent space
    if noise is None:
        latent_samples = get_noise(model, n_samples)
    else:
        latent_samples = noise

    # generate the images
    generated_images = model.generator(latent_samples)

    # create directory if not exists
    if not os.path.exists('images'):
        os.makedirs('images')

    # save the images to disk
    image_filenames = []
    for i in range(n_samples):
        image = generated_images[i]
        image = tf.cast(image, tf.float32)
        image_filename = f'images/{prefix}{i}.png'
        if not os.path.exists(os.path.dirname(image_filename)):
            os.makedirs(os.path.dirname(image_filename))
        tf.keras.preprocessing.image.save_img(image_filename, image, data_format='channels_last',
                                            file_format='png', scale=True)
        
