import tensorflow as tf


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
                 generation_shape=(28,28),
                 output_classes=2):
        super().__init__()
        self.input_shape = generation_shape
        self.output_shape = (output_classes,)

        self.feedforward = tf.keras.models.Sequential([
            #TODO define layers
            tf.keras.layers.Dense(self.output_shape),
        ])

    def call(self, x, training=False):
        #TODO does this have to be in the context of a GradientTape if training?
        return self.feedforward(x)
