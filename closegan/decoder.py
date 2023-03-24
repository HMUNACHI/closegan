import tensorflow as tf
from keras import layers

class ConditionalTransformerDecoder(layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim, latent_dim, n_blocks=1, heads=3):
        super().__init__()
        self.embeddings = TokenAndPositionEmbedding(max_len, vocab_size, embedding_dim)
        self.encoder_decoder_attention = [layers.MultiHeadAttention(heads, embedding_dim) for _ in range(n_blocks)]
        self.transformers = [TransformerBlock(embedding_dim, heads, latent_dim) for _ in range(n_blocks)]
        self.dense = layers.Dense(vocab_size)
 
    def call(self, encoder_output, conditions):
        conditions_embeddings = self.embeddings(conditions)

        for transformerBlock, attentionBlock in zip(self.transformers, self.encoder_decoder_attention):
            conditioned_z = attentionBlock(conditions_embeddings, encoder_output)
            decoder_outputs = transformerBlock(conditioned_z)

        final_outputs = self.dense(decoder_outputs)
        return [final_outputs, decoder_outputs]


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, max_len, vocab_size, embedding_dim):
        super().__init__()
        self.token_embeddings = layers.Embedding(input_dim=vocab_size, output_dim=embedding_dim)
        self.position_embeddings = layers.Embedding(input_dim=max_len, output_dim=embedding_dim)

    def call(self, inputs):
        maxlen = tf.shape(inputs)[-1]
        tokens = self.token_embeddings(inputs)
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.position_embeddings(positions)
        return tokens + positions


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, latent_dim, dropout=0.1):
        super().__init__()
        self.attention = layers.MultiHeadAttention(num_heads, embed_dim)
        self.dense1 = layers.Dense(latent_dim, activation="relu")
        self.dense2 = layers.Dense(embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)

    def causal_attention_mask(self, batch_size, destination_dim, source_dim, dtype):
        idx_source = tf.range(destination_dim)[:, None]
        idx_destination = tf.range(source_dim)
        mask = idx_source >= idx_destination - source_dim + destination_dim
        mask = tf.cast(mask, dtype)
        mask = tf.reshape(mask, [1, destination_dim, source_dim])
        concatenator = tf.concat([tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],0)
        return tf.tile(mask, concatenator)

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        sequence_length = input_shape[1]
        causal_mask = self.causal_attention_mask(batch_size, sequence_length, sequence_length, tf.bool)
        attention_output = self.attention(inputs, inputs, attention_mask=causal_mask)
        attention_output = self.dropout1(attention_output)
        normalised_output = self.layernorm1(inputs + attention_output)
        dense_output1 = self.dense1(normalised_output)
        dense_output2 = self.dense1(dense_output1)
        dense_output = self.dropout2(dense_output2)
        return self.layernorm2(normalised_output + dense_output)