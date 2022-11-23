import tensorflow as tf


SEED = 8675309


class Generator(tf.keras.Model):
    def __init__(self,
                 latent_shape=(100,),
                 generation_shape=(28,28)):
        super().__init__()
        self.input_shape = latent_shape
        self.output_shape = generation_shape

        self.feedforward = tf.keras.models.Sequential([
            #TODO define layers
            tf.keras.layers.Dense(self.output_shape),
        ])
        

    def call(self, x, training=False):
        #TODO does this have to be in the context of a GradientTape if training?
        return self.feedforward(x)


class Discriminator(tf.keras.Model):
    def __init__(self,
                 generation_shape=(28, 28),
                 return_logits=True):
        super().__init__()
        self.input_shape = generation_shape
        self.output_shape = (1,)

        self.return_logits = return_logits
        final_activation = None if return_logits else 'softmax'

        self.feedforward = tf.keras.models.Sequential([
            #TODO define layers
            tf.keras.layers.Dense(self.output_shape, activation=final_activation),
        ])

    def call(self, x, training=False):
        #TODO does this have to be in the context of a GradientTape if training?
        # answer is potentially no beacuse the gradient tape is in GAN.train_step()
        return self.feedforward(x)


class GAN(tf.keras.Model):
    def __init__(self,
                 generator,
                 discriminator,
                 noise_generator='normal'):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator

        self.generator_loss     = tf.keras.losses.BinaryCrossentropy(from_logits=discriminator.return_logits)
        self.discriminator_loss = tf.keras.losses.BinaryCrossentropy(from_logits=discriminator.return_logits)

        match noise_generator:
            case 'normal':
                self.noise_generator_helper = tf.random.Generator.from_seed(SEED)
                self.noise_generator = lambda batch_size: self.noise_generator_helper.normal([batch_size,
                    *self.generator.input_shape])
            case _:
                raise NotImplementedError


    def compile(self,
                generator_optimizer='adam',
                discriminator_optimizer='adam'):
        super().compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer


    def train_step(self, x):
        # x is real data
        batch_size = x.shape[0]

        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            # generation
            noise = self.noise_generator(batch_size)
            generator_output = self.generator(noise, training=True)  #TODO is training=True needed?

            # discrimination preprocessing
            discriminator_input = tf.concat([x, generator_output], 0)
            discriminator_labels = tf.concat([tf.zeros(batch_size),    # real
                                              tf.ones(batch_size)])    # fake

            # discriminator input/label shuffling
            shuffled_indices = tf.random.shuffle(tf.range(2 * batch_size))
            discriminator_input  = tf.gather(discriminator_input, shuffled_indices)
            discriminator_labels = tf.gather(discriminator_labels, shuffled_indices)

            # discrimination
            discriminator_output = self.discriminator(discriminator_input, training=True)

            # discriminator loss
            discriminator_loss = self.discriminator_loss(discriminator_labels, discriminator_output)
            
            # generator loss
            generator_labels = tf.zeros(batch_size)  # generator is looking for zeros (incorrect) output from discriminator
            discriminator_outputs_from_generator = tf.gather(discriminator_output,
                                                             tf.where(discriminator_labels)) # indices where correct answer was 1 (fake)
            generator_loss = self.generator_loss(generator_labels, discriminator_outputs_from_generator)

        # generator update
        generator_grads = generator_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))
        # discriminator update
        discriminator_grads = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_weights))

