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

        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(tf.math.reduce_prod(self.generation_shape), activation='tanh'),  #TODO don't hardcode
            tf.keras.layers.Reshape(self.generation_shape)
        ])
        

    def call(self, x, training=False) -> tf.Tensor:
        #TODO does this have to be in the context of a GradientTape if training?
        output = self.feedforward(x)
        if not isinstance(output, tf.Tensor):
            raise
        return output

class Discriminator(tf.keras.Model):
    generation_shape = (32, 32, 3)

    def __init__(self):
        super().__init__()

        final_activation = None if FROM_LOGITS else 'sigmoid'

        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.01),
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

            # accuracies
            # discriminator_accuracy = tf.reduce_mean(
            #     tf.keras.metrics.binary_accuracy(discriminator_labels, discriminator_output))
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

    def save(self, filepath):
        # delete filepath if exists
        if os.path.exists(filepath):
            shutil.rmtree(filepath)
        # create empty directory at filepath
        os.mkdir(filepath)

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
