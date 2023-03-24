import tensorflow as tf
from keras import layers, backend
from keras.losses import sparse_categorical_crossentropy

class VariationalBiLSTMEncoder(layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim, latent_dim, dropout=0.1):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, trainable=True)
        self.biLSTM = layers.Bidirectional(layers.LSTM(latent_dim, dropout=dropout))
        self.mean = layers.Dense(latent_dim)
        self.std = layers.Dense(latent_dim)
        self.sampling = Sampling()

    def call(self, inputs):
        embedding = self.embedding(inputs)
        contextual_encodings = self.biLSTM(embedding)
        mean = self.mean(contextual_encodings)
        std = self.std(contextual_encodings)
        variational_latent_vector = self.sampling([mean, std])
        return variational_latent_vector


class Sampling(layers.Layer):
    def call(self, inputs):
        z_mean, z_std = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_std) * epsilon