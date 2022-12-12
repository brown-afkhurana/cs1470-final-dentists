import tensorflow as tf
import numpy as np
import os


os.environ["CUDA_VISIBLE_DEVICES"] = ''


def load_cifar_images():
    if os.path.exists('data/cifar_fake'):
        print('Loading tensors')
        np_images = np.load('data/cifar_fake/images.npy')
        np_labels = np.load('data/cifar_fake/labels.npy')
        return tf.convert_to_tensor(np_images), tf.convert_to_tensor(np_labels)
        # return tf.io.read_file('data/cifar_fake/images'), tf.io.read_file('data/cifar_fake/labels')

    root_directory = 'images/cifar'
    all_images = []
    all_labels = []
    for i in range(10):
        print(f'i={i}')
        digit_directory = f'{root_directory}/{i}/'
        filenames = os.listdir(digit_directory)
        filenames.sort()
        pil_images = []
        for filename in filenames[2000:]:
            try:
                pil_images.append(tf.keras.utils.load_img(f'{digit_directory}/{filename}', color_mode='rgb'))
            except FileNotFoundError:
                breakpoint()
        # pil_images = [tf.keras.utils.load_img(filename, grayscale=True) for filename in filenames]
        array_images = [tf.keras.utils.img_to_array(image) for image in pil_images]
        all_images += (array_images)
        all_labels += [i for _ in array_images]

    np_images = np.asarray(all_images, dtype=np.float32)
    images = tf.convert_to_tensor(np_images, dtype=tf.float32)
    np_labels = np.asarray(all_labels, dtype=np.float32)
    labels = tf.convert_to_tensor(np_labels, dtype=tf.float32)

    print('Saving tensors')
    # tf.io.write_file('data/cifar_fake/images', images)
    os.makedirs('data/cifar_fake')
    np.save('data/cifar_fake/images.npy', np_images)
    print('Saved images')
    # tf.io.write_file('data/cifar_fake/labels', labels)
    np.save('data/cifar_fake/labels.npy', np_labels)
    print('Saved labels')

    return images, labels


def train_classifier(train_images, train_labels, test_images, test_labels, epochs=16):
    classifier = tf.keras.Sequential([
        tf.keras.layers.Dropout(0.1),
        tf.keras.layers.Conv2D(32, (3, 3), strides=(1, 1), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding="same"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256),
        tf.keras.layers.Dense(10),
    ])

    classifier.compile(optimizer=tf.keras.optimizers.Adam(),
                       loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    classifier.fit(train_images, train_labels, epochs=epochs, validation_split=0.1, batch_size=128)


    results = classifier.evaluate(test_images, test_labels, batch_size=128)
    print(results)

    return classifier, results


def train_cifar_classifier_real():
    print('CIFAR10 REAL')
    (train_img, train_lab), (test_img, test_lab) = tf.keras.datasets.cifar10.load_data()

    train_img = tf.image.resize(train_img, [32,32])
    test_img = tf.image.resize(test_img, [32, 32])

    train_img = train_img / 255.
    test_img = test_img / 255.

    shuffled_indices = tf.random.shuffle(tf.range(train_img.shape[0]))
    # shuffled_indices = shuffled_indices[:60000]
    train_img  = tf.gather(train_img, shuffled_indices)
    train_lab = tf.gather(train_lab, shuffled_indices)

    print(train_img.shape)
    print(train_lab.shape)

    classifier, results = train_classifier(train_img, train_lab, test_img, test_lab)

    return classifier


def train_cifar_classifier_fake():
    print('CIFAR10 FAKE')
    images, labels = load_cifar_images()
    print(images.shape)
    print(labels.shape)

    (_, _), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    images = images / 255.
    test_images = test_images / 255.

    shuffled_indices = tf.random.shuffle(tf.range(images.shape[0]))
    shuffled_indices = shuffled_indices[:60000]
    images = tf.gather(images, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)

    test_images = tf.image.resize(test_images, [32, 32])

    classifier, results = train_classifier(images, labels, test_images, test_labels)

    return classifier


def train_cifar_classifier_combined():
    print('CIFAR10 COMBINED')
    fake_images, fake_labels = load_cifar_images()
    (real_images, real_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()

    # fake_images = tf.expand_dims(fake_images, -1)
    # real_images = tf.expand_dims(real_images, -1)
    # test_images = tf.expand_dims(test_images, -1)

    fake_images = tf.cast(fake_images, tf.float32)
    real_images = tf.cast(real_images, tf.float32)
    test_images = tf.cast(test_images, tf.float32)

    fake_images = fake_images / 255.0
    real_images = real_images / 255.0
    test_images = test_images / 255.0

    real_images = tf.image.resize(real_images, [32, 32])
    test_images = tf.image.resize(test_images, [32, 32])

    shuffled_indices = tf.random.shuffle(tf.range(fake_images.shape[0]))
    fake_images = tf.gather(fake_images, shuffled_indices)
    fake_labels = tf.gather(fake_labels, shuffled_indices)

    images = tf.concat([fake_images, real_images], 0)
    labels = tf.concat([fake_labels, real_labels], 0)

    shuffled_indices = tf.random.shuffle(tf.range(images.shape[0]))
    images = tf.gather(images, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)

    classifier, results = train_classifier(images, labels, test_images, test_labels, epochs=8)

    return classifier


if __name__ == '__main__':
    # load_cifar_images()
    train_cifar_classifier_real()
    train_cifar_classifier_fake()
    train_cifar_classifier_combined()
