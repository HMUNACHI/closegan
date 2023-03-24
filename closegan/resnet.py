import tensorflow as tf
from keras import layers

class LinearResNet2D(layers.Layer):
    def __init__(self, latent_dim, blocks=1):
        super().__init__()
        self.blocks = [ResidualBlock(latent_dim) for _ in range(blocks)]

    def call(self, inputs):
        for block in self.blocks:
            inputs = block(inputs)
        return inputs

class ResidualBlock(layers.Layer):
    def __init__(self, latent_dim):
        super(ResidualBlock, self).__init__()
        self.dense1 = layers.Dense(latent_dim, activation="relu")
        self.dense2 = layers.Dense(latent_dim)
    
    def call(self, inputs):
        outputs = self.dense1(inputs)
        outputs = self.dense2(outputs)
        return outputs + inputs