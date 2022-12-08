import gzip

import numpy as np
import tensorflow as tf

#reshape to 32,32 the same way we do in HW6 
#duplicate last axis to R,G,B but have them be all the same color since black and white
#push

def get_data_MNIST(subset, data_path="../data"):

    ## http://yann.lecun.com/exdb/mnist/
    subset = subset.lower().strip()
    assert subset in ("test", "train"), f"unknown data subset {subset} requested"
    inputs_file_path, labels_file_path, num_examples = {
        "train": ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz", 60000),
        "test": ("t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz", 10000),
    }[subset]
    inputs_file_path = f"{data_path}/mnist/{inputs_file_path}"
    labels_file_path = f"{data_path}/mnist/{labels_file_path}"


    ##Declare image variable
    image = []

    with open(inputs_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream: 
        ##trash (headers ignore)
        header = bytestream.read(16)
        ##actual image (num exmaple bytes )
 
        ##convert to np.uint8

        image_to_shape = np.frombuffer(bytestream.read(), np.uint8)
        image_casted = []
        for i in image_to_shape: 
            image_casted.append(np.float32(i)) 

        new_list= [x/255.0 for x in image_casted]
            
        image = np.reshape(new_list, (num_examples,784))


        #image_nonnormalized = np.frombuffer(bytestream.read(), np.uint8)
        #image_non_flat = [i/255.0 for i in image_nonnormalized]
        #image = np.reshape(np.float32(image_non_flat), (num_examples,784))
    

    label = []

    with open(labels_file_path, 'rb') as f, gzip.GzipFile(fileobj=f) as bytestream: 
        ##trash (headers ignore)
        header = bytestream.read(8)

        ##actual image (num exmaple bytes )

        ##convert to np.uint8
        label = np.frombuffer(bytestream.read(), np.uint8)

    return image, label


def get_data_CIFAR(subset,
                   return_one_hot = True):
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.cifar10.load_data()
    #50,000 training images and 10,000 test images

    depth = 10

    # image resizing is unecessary, so it is commented out
    # RGB values are normalized between 0 and 1
    # labels are converted into one hot tensors
    if subset == "train":
        #train_image = tf.expand_dims(train_image, axis=-1)
        #train_image = tf.image.resize(train_image, [32,32,3])
        image = train_image/255
        if return_one_hot:
            label = tf.one_hot(train_label, depth)
        else:
            label = train_label


    if subset == "test":
        # test_image = tf.expand_dims(test_image, axis=-1)
        # test_image = tf.image.resize(test_image, [32,32,3])
        image = test_image/255
        if return_one_hot:
            label = tf.one_hot(test_label, depth)
        else:
            label = test_label

    return image, label
