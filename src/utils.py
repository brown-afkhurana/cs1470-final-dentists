import tensorflow as tf
import imageio

def generate_and_save_images(model, n_samples=5):
    # generate 5 samples of the latent space
    latent_samples = model.get_noise(n_samples)

    # generate the images
    generated_images = model.generator(latent_samples)

    # save the images to disk
    image_filenames = []
    for i in range(n_samples):
        image = generated_images[i]
        image = tf.cast(image, tf.uint8)
        image_filename = f'images/{i}.png'
        tf.keras.preprocessing.image.save_img(image_filename, image)
