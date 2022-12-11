import gzip
import tensorflow as tf

import numpy as np

def get_data_MNIST(subset, return_one_hot=True):
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
    #60,000 training images and 10,000 test images

    depth = 10

    #image resized to 32x32
    #Grayscale values are normalized between 0 and 1
    #labels are converted into one hot tensors
    if subset == "train":
        train_image = tf.cast(train_image, tf.float32)
        train_image = tf.expand_dims(train_image, axis=-1)
        train_image = tf.image.resize(train_image, [32,32])
        image = train_image/255
        if return_one_hot:
            label = tf.one_hot(train_label, depth)
        else:
            label = train_label

    if subset == "test":
        test_image = tf.cast(test_image, tf.float32)
        test_image = tf.expand_dims(test_image, axis=-1)
        test_image = tf.image.resize(test_image, [32,32])
        image = test_image/255
        if return_one_hot:
            label = tf.one_hot(test_label, depth)
        else:
            label = test_label

    return image, label

def get_data_CIFAR(subset, return_one_hot = True):
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.cifar10.load_data()
    #50,000 training images and 10,000 test images

    depth = 10

    # image resizing is unecessary, so it is commented out
    # RGB values are normalized between 0 and 1
    # labels are converted into one hot tensors
    if subset == "train":
        train_image = tf.cast(train_image, tf.float32)
        #train_image = tf.expand_dims(train_image, axis=-1)
        #train_image = tf.image.resize(train_image, [32,32,3])
        image = train_image/255
        if return_one_hot:
            label = tf.one_hot(train_label, depth)
        else:
            label = train_label


    if subset == "test":
        test_image = tf.cast(test_image, tf.float32)
        # test_image = tf.expand_dims(test_image, axis=-1)
        # test_image = tf.image.resize(test_image, [32,32,3])
        image = test_image/255
        if return_one_hot:
            label = tf.one_hot(test_label, depth)
        else:
            label = test_label

    return image, label

def get_data_CIFAR10H(subset, return_one_hot = True):  # subset is a null variable
    #cifar10h-probs.npy should be stored in the data package
    
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.cifar10.load_data()
    #10,000 images from the test set

    depth = 10

    # image resizing is unecessary, so it is commented out
    # RGB values are normalized between 0 and 1
    # labels are converted into one hot tensors
    test_image = tf.cast(test_image, tf.float32)
    # test_image = tf.expand_dims(test_image, axis=-1)
    # test_image = tf.image.resize(test_image, [32,32,3])
    image = test_image/255.0

    # CIFAR10H_file_path = '../data/cifar10h-probs.npy'
    CIFAR10H_file_path = 'data/cifar10h-probs.npy'
    probability_labels = np.load(CIFAR10H_file_path)

    # convert the 10000x10 array into a 10000 element list of labels
    # for probability_distribution in probability_labels:
    #     image_label = tf.argmax(probability_distribution, axis=-1)
    #     probability_distribution = image_label
    probability_labels = tf.argmax(probability_labels, axis=-1)

    if return_one_hot:
        label = tf.one_hot(probability_labels, depth)
    else:
        label = probability_labels
        

    return image, label

def get_data_CIFAR10H_counts(subset, return_one_hot = True):  # subset is a null variable
    #cifar10h-probs.npy should be stored in the data package
    
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.cifar10.load_data()
    #10,000 images from the test set

    depth = 10

    # image resizing is unecessary, so it is commented out
    # RGB values are normalized between 0 and 1
    # labels are converted into one hot tensors
    test_image = tf.cast(test_image, tf.float32)
    # test_image = tf.expand_dims(test_image, axis=-1)
    # test_image = tf.image.resize(test_image, [32,32,3])
    image = test_image/255.0

    # CIFAR10H_file_path = '../data/cifar10h-probs.npy'
    CIFAR10H_file_path = 'data/cifar10h-counts.npy'
    count_labels = np.load(CIFAR10H_file_path)

    # convert the 10000x10 array into a 10000 element list of labels
    # for probability_distribution in probability_labels:
    #     image_label = tf.argmax(probability_distribution, axis=-1)
    #     probability_distribution = image_label
    images = []
    labels = []
    for i, labels_counts in enumerate(count_labels):
        for label, count in enumerate(labels_counts):
            images += [image[i]] * count
            labels += [label] * count

    image = np.array(images)
    label = np.array(labels)
    if return_one_hot:
        label = tf.one_hot(label, depth)
    else:
        label = label
        
    image = tf.convert_to_tensor(image)
    label = tf.convert_to_tensor(label)

    print(image.shape, label.shape)

    return image, label
