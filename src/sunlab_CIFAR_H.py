import tensorflow as tf
import numpy as np
import os

from preprocessing import get_data_CIFAR10H, get_data_CIFAR10H_probs


os.environ["CUDA_VISIBLE_DEVICES"] = ''


def load_cifar_h_images():
    if os.path.exists('data/cifar_h_fake'):
        print('Loading tensors')
        np_images = np.load('data/cifar_h_fake/images.npy')
        np_labels = np.load('data/cifar_h_fake/labels.npy')
        return tf.convert_to_tensor(np_images), tf.convert_to_tensor(np_labels)
        # return tf.io.read_file('data/cifar_h_fake/images'), tf.io.read_file('data/cifar_h_fake/labels')

    root_directory = 'images/cifar10h_counts/'
    all_images = []
    all_labels = []
    for i in range(10):
        print(f'i={i}')
        digit_directory = f'{root_directory}/{i}/'
        filenames = os.listdir(digit_directory)
        filenames.sort()
        pil_images = []
        for filename in filenames[100:]:  # skip first quarter
            try:
                pil_images.append(tf.keras.utils.load_img(f'{digit_directory}/{filename}', color_mode='rgb'))
            except FileNotFoundError:
                breakpoint()
        array_images = [tf.keras.utils.img_to_array(image) for image in pil_images]
        all_images += (array_images)
        all_labels += [i for _ in array_images]

    np_images = np.asarray(all_images, dtype=np.float32)
    images = tf.convert_to_tensor(np_images, dtype=tf.float32)
    np_labels = np.asarray(all_labels, dtype=np.float32)
    labels = tf.convert_to_tensor(np_labels, dtype=tf.float32)

    print('Saving tensors')
    # tf.io.write_file('data/cifar_h_fake/images', images)
    os.makedirs('data/cifar_h_fake')
    np.save('data/cifar_h_fake/images.npy', np_images)
    print('Saved images')
    # tf.io.write_file('data/cifar_h_fake/labels', labels)
    np.save('data/cifar_h_fake/labels.npy', np_labels)
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
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    classifier.compile(optimizer=tf.keras.optimizers.Adam(),
                    #    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                       loss=tf.keras.losses.MeanSquaredError(),
                    #    loss=tf.keras.losses.KLDivergence(),
                       metrics=[tf.keras.metrics.CategoricalAccuracy()])

    classifier.fit(train_images, train_labels, epochs=epochs, validation_split=0.1, batch_size=128)


    results = classifier.evaluate(test_images, test_labels, batch_size=128)
    print(results)

    return classifier, results


def train_cifar_h_classifier_real():
    print('CIFAR10-H REAL')
    (test_images, test_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    train_images, train_labels = get_data_CIFAR10H_probs()

    test_labels = tf.one_hot(test_labels, 10)

    train_images = train_images / 255.
    test_images = test_images / 255.

    shuffled_indices = tf.random.shuffle(tf.range(train_images.shape[0]))
    shuffled_indices = shuffled_indices[:3000]  # match 300 per class
    train_images = tf.gather(train_images, shuffled_indices)
    train_labels = tf.gather(train_labels, shuffled_indices)

    print(train_images.shape)
    print(train_labels.shape)

    classifier, results = train_classifier(train_images, train_labels, test_images, test_labels)

    return classifier


def train_cifar_h_classifier_fake():
    print('CIFAR10-H FAKE')
    images, labels = load_cifar_h_images()
    print(images.shape)
    print(labels.shape)

    labels = tf.cast(labels, tf.uint8)

    (test_images, test_labels), (_, _) = tf.keras.datasets.cifar10.load_data()

    labels = tf.one_hot(labels, 10)
    test_labels = tf.one_hot(test_labels, 10)
    test_labels = tf.squeeze(test_labels)

    images = images / 255.
    test_images = test_images / 255.

    shuffled_indices = tf.random.shuffle(tf.range(images.shape[0]))
    images = tf.gather(images, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)

    classifier, results = train_classifier(images, labels, test_images, test_labels)

    return classifier


def train_cifar_h_classifier_combined():
    print('CIFAR10-H COMBINED')
    fake_images, fake_labels = load_cifar_h_images()
    fake_labels = tf.cast(fake_labels, tf.uint8)
    fake_labels = tf.one_hot(fake_labels, 10)
    (test_images, test_labels), (_, _) = tf.keras.datasets.cifar10.load_data()
    test_labels = tf.one_hot(test_labels, 10, dtype=tf.float32)

    real_images, real_labels = get_data_CIFAR10H_probs()
    shuffled_indices = tf.random.shuffle(tf.range(real_images.shape[0]))
    shuffled_indices = shuffled_indices[:3000]  # match 300 per class
    real_images = tf.gather(real_images, shuffled_indices)
    real_labels = tf.gather(real_labels, shuffled_indices)

    real_labels = tf.cast(real_labels, tf.float32)

    fake_images = fake_images / 255.0
    real_images = real_images / 255.0
    test_images = test_images / 255.0

    images = tf.concat([fake_images, real_images], 0)
    labels = tf.concat([fake_labels, real_labels], 0)

    shuffled_indices = tf.random.shuffle(tf.range(images.shape[0]))
    images = tf.gather(images, shuffled_indices)
    labels = tf.gather(labels, shuffled_indices)

    test_labels = tf.squeeze(test_labels)

    classifier, results = train_classifier(images, labels, test_images, test_labels, epochs=8)

    return classifier


if __name__ == '__main__':
    # load_cifar_h_images()
    # train_cifar_h_classifier_real()
    # train_cifar_h_classifier_fake()
    train_cifar_h_classifier_combined()
