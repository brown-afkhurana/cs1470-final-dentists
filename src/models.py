import os
import shutil

import tensorflow as tf


SEED = 8675309

FROM_LOGITS = True


class Generator(tf.keras.Model):

    latent_shape = (100,)
    generation_shape = (32, 32, 3)

    def __init__(self):
        super().__init__()

        '''
        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(tf.math.reduce_prod(self.generation_shape), activation='tanh'),
            tf.keras.layers.Reshape(self.generation_shape)
        ])
        '''
        channels = self.generation_shape[-1]
        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Reshape((4, 4, 16)),
            tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(0.1),
            tf.keras.layers.Conv2DTranspose(channels, (3, 3), strides=(1, 1), padding='same'),
            tf.keras.layers.Activation('sigmoid'),
        ])
        

    def call(self, x, training=False) -> tf.Tensor:
        #TODO does this have to be in the context of a GradientTape if training?
        output = self.feedforward(x)
        if not isinstance(output, tf.Tensor):
            raise
        return output

class Discriminator(tf.keras.Model):
    generation_shape = (32, 32, 3)

    def __init__(self, with_noise=False):
        super().__init__()

        final_activation = None if FROM_LOGITS else 'sigmoid'

        if with_noise:
            noise_layer = [tf.keras.layers.GaussianNoise(0.1)]
        else:
            noise_layer = []
    
        '''
        self.feedforward = tf.keras.Sequential(noise_layer + [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(1, activation=final_activation),
        ])
        '''
        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same'),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation=final_activation),
        ])

    def call(self, x, training=False) -> tf.Tensor:
        output = self.feedforward(x)
        if not isinstance(output, tf.Tensor):
            raise
        return output


class GAN(tf.keras.Model):
    def __init__(self,
                 generator: tf.keras.Model,
                 discriminator: tf.keras.Model,
                 noise_shape='normal',
                 seed=SEED):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.generator_loss     = tf.keras.losses.BinaryCrossentropy(from_logits=FROM_LOGITS)
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=FROM_LOGITS)

        self.seed = seed
        self.noise_generator = tf.random.Generator.from_seed(seed)

        self.noise_shape = noise_shape

    def call(self, inputs) -> tf.Tensor:
        output = self.generator(inputs)
        if not isinstance(output, tf.Tensor):
            raise
        return output

    def get_noise(self, batch_size):
        match self.noise_shape:
            case 'normal':
                return self.noise_generator.normal(shape=(batch_size, *Generator.latent_shape))
            case _:
                raise NotImplementedError(f'Noise shape {self.noise_shape} not implemented')

    def compile(self,
                generator_optimizer=tf.keras.optimizers.Adam(),
                discriminator_optimizer=tf.keras.optimizers.Adam()):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer


    def train_step(self, x):
        # x is real data
        x, _ = x
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # generation
            noise = self.noise_generator.normal(shape=(batch_size, *Generator.latent_shape))
            generator_output = self.generator(noise)

            # discrimination preprocessing
            discriminator_input = tf.concat([x, generator_output], 0)
            discriminator_labels = tf.concat([tf.zeros(batch_size),    # real
                                              tf.ones(batch_size)],    # fake
                                              0)

            # discriminator input/label shuffling
            shuffled_indices = tf.random.shuffle(tf.range(2 * batch_size))
            discriminator_input  = tf.gather(discriminator_input, shuffled_indices)
            discriminator_labels = tf.gather(discriminator_labels, shuffled_indices)

            # noise for discriminator
            """
            noise = self.noise_generator.normal(shape=(tf.shape(discriminator_input)))
            noise = noise * self.discriminator_leading  # if generator leading, 0 noise
            discriminator_input = discriminator_input + noise
            """

            # discrimination
            discriminator_output = self.discriminator(discriminator_input)

            # discriminator loss
            discriminator_loss = self.discriminator_loss(discriminator_labels, discriminator_output)
            
            # generator loss
            generator_labels = tf.zeros(batch_size)  # generator is looking for zeros (incorrect) output from discriminator
            discriminator_outputs_from_generator = tf.gather(discriminator_output,
                                                             tf.where(discriminator_labels)) # indices where correct answer was 1 (fake)
            discriminator_outputs_from_data      = tf.gather(discriminator_output,
                                                             tf.where(
                                                                tf.equal(discriminator_labels, 0))) # indices where correct answer was 0 (real)
            discriminator_outputs_from_generator = tf.squeeze(discriminator_outputs_from_generator)
            discriminator_outputs_from_data      = tf.squeeze(discriminator_outputs_from_data)
            generator_loss = self.generator_loss(generator_labels, discriminator_outputs_from_generator)
            # blackscreen loss
            # generator_loss += tf.reduce_mean(tf.reduce_max(generator_output, axis=(1, 2, 3)))

            # accuracies
            # discriminator_accuracy = tf.reduce_mean(
            #     tf.keras.metrics.binary_accuracy(discriminator_labels, discriminator_output))
            discriminator_accuracy_fake = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(tf.ones(batch_size), discriminator_outputs_from_generator))
            discriminator_accuracy_real = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(tf.zeros(batch_size), discriminator_outputs_from_data))
            generator_accuracy = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(generator_labels, discriminator_outputs_from_generator))

            # self.discriminator_leading = tf.math.sigmoid(discriminator_accuracy_fake - generator_accuracy)

        # generator update
        generator_grads = generator_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))
        # discriminator update
        discriminator_grads = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_weights))

        return {'G_loss': generator_loss, 
                'D_loss': discriminator_loss,
                'G_acc': generator_accuracy,
                'D_acc_R': discriminator_accuracy_real,
                'D_acc_F': discriminator_accuracy_fake}

    def save(self, filepath):
        # delete filepath if exists
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        # create empty directory at filepath
        os.makedirs(filepath)

        # save generator
        self.generator.save(os.path.join(filepath, 'generator'))
        # save discriminator
        self.discriminator.save(os.path.join(filepath, 'discriminator'))

    @classmethod
    def load(cls, filepath):
        # load generator
        generator = tf.keras.models.load_model(os.path.join(filepath, 'generator'))
        # load discriminator
        discriminator = tf.keras.models.load_model(os.path.join(filepath, 'discriminator'))

        if not (isinstance(generator, tf.keras.Model) and isinstance(discriminator, tf.keras.Model)):
            raise  # for typing purposes

        # typechecking
        if not (isinstance(generator, tf.keras.Model) and isinstance(discriminator, tf.keras.Model)):
            raise TypeError('Generator and discriminator must be instances of tf.keras.Model')

        return cls(generator,
                   discriminator,
                   seed=SEED)

class CGAN(GAN):
    def train_step(self, x):
        # x is real data
        x, labels = x
        x = tf.cast(x, tf.float32)
        batch_size = tf.shape(x)[0]

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # generation
            noise = self.noise_generator.normal(shape=(batch_size, *Generator.latent_shape))
            # add conditional label to noise
            gen_labels_to_concat = tf.tile(labels[:, None],
                                           [1, self.generator.latent_shape[0]])
            gen_labels_to_concat = tf.cast(gen_labels_to_concat, tf.float32)
            generator_input = tf.concat([noise, gen_labels_to_concat], 1)  # this is incorrect
            generator_output = self.generator(generator_input)

            # discrimination preprocessing
            discriminator_input = tf.concat([x, generator_output], 0)
            discriminator_labels = tf.concat([tf.zeros(batch_size),    # real
                                              tf.ones(batch_size)],    # fake
                                              0)

            # discriminator input/label shuffling
            shuffled_indices = tf.random.shuffle(tf.range(2 * batch_size))
            discriminator_input  = tf.gather(discriminator_input, shuffled_indices)
            discriminator_labels = tf.gather(discriminator_labels, shuffled_indices)

            # noise for discriminator
            """
            noise = self.noise_generator.normal(shape=(tf.shape(discriminator_input)))
            noise = noise * self.discriminator_leading  # if generator leading, 0 noise
            discriminator_input = discriminator_input + noise
            """
            # add conditional label to discriminator input
            disc_labels_to_concat = tf.tile(labels[:, None, None, None],
                                            [1, *(self.discriminator.generation_shape[:-1]), 1])
            disc_labels_to_concat = tf.cast(disc_labels_to_concat, tf.float32)
            discriminator_input = tf.concat([discriminator_input, disc_labels_to_concat], -1)

            # discrimination
            discriminator_output = self.discriminator(discriminator_input)

            # discriminator loss
            discriminator_loss = self.discriminator_loss(discriminator_labels, discriminator_output)
            
            # generator loss
            generator_labels = tf.zeros(batch_size)  # generator is looking for zeros (incorrect) output from discriminator
            discriminator_outputs_from_generator = tf.gather(discriminator_output,
                                                             tf.where(discriminator_labels)) # indices where correct answer was 1 (fake)
            discriminator_outputs_from_data      = tf.gather(discriminator_output,
                                                             tf.where(
                                                                tf.equal(discriminator_labels, 0))) # indices where correct answer was 0 (real)
            discriminator_outputs_from_generator = tf.squeeze(discriminator_outputs_from_generator)
            discriminator_outputs_from_data      = tf.squeeze(discriminator_outputs_from_data)
            generator_loss = self.generator_loss(generator_labels, discriminator_outputs_from_generator)

            discriminator_accuracy_fake = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(tf.ones(batch_size), discriminator_outputs_from_generator))
            discriminator_accuracy_real = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(tf.zeros(batch_size), discriminator_outputs_from_data))
            generator_accuracy = tf.reduce_mean(
                tf.keras.metrics.binary_accuracy(generator_labels, discriminator_outputs_from_generator))


        # generator update
        generator_grads = generator_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))
        # discriminator update
        discriminator_grads = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_weights))

        return {'G_loss': generator_loss, 
                'D_loss': discriminator_loss,
                'G_acc': generator_accuracy,
                'D_acc_R': discriminator_accuracy_real,
                'D_acc_F': discriminator_accuracy_fake}