import tensorflow as tf


SEED = 8675309


class Generator(tf.keras.Model):
    def __init__(self,
                 latent_shape=(100,),
                 generation_shape=(28,28)):
        super().__init__()
        self.hardcode_input_shape = latent_shape
        self.hardcode_output_shape = generation_shape

        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(1000),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(tf.math.reduce_prod(self.hardcode_output_shape), activation='tanh'),  #TODO don't hardcode
            tf.keras.layers.Reshape(self.hardcode_output_shape)
        ])
        

    def call(self, x, training=False):
        #TODO does this have to be in the context of a GradientTape if training?
        return self.feedforward(x)


class Discriminator(tf.keras.Model):
    def __init__(self,
                 generation_shape=(28, 28, 1),
                 return_logits=True):
        super().__init__()
        self.hardcode_input_shape = generation_shape
        self.hardcode_output_shape = (1,)

        self.return_logits = return_logits
        final_activation = None if return_logits else 'sigmoid'

        self.feedforward = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(256),
            tf.keras.layers.LeakyReLU(0.01),
            tf.keras.layers.Dense(1, activation=final_activation),
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
                self.noise_generator = tf.random.Generator.from_seed(SEED)
            case _:
                raise NotImplementedError


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
            noise = self.noise_generator.normal(shape=(batch_size, *self.generator.hardcode_input_shape))
            generator_output = self.generator(noise)  #TODO is training=True needed?

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
            # print(discriminator_outputs_from_generator.shape)
            discriminator_outputs_from_generator = tf.squeeze(discriminator_outputs_from_generator)
            generator_loss = self.generator_loss(generator_labels, discriminator_outputs_from_generator)

        # generator update
        generator_grads = generator_tape.gradient(generator_loss, self.generator.trainable_weights)
        self.generator_optimizer.apply_gradients(zip(generator_grads, self.generator.trainable_weights))
        # discriminator update
        discriminator_grads = discriminator_tape.gradient(discriminator_loss, self.discriminator.trainable_weights)
        self.discriminator_optimizer.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_weights))

        return {'G_loss': generator_loss, 'D_loss': discriminator_loss}

