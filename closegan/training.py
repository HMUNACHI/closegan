import keras
import tensorflow as tf

from model import CloseGAN
from resnet import LinearResNet2D
from transformers import TFBertModel
from config import embedding_dim, hidden_dim, bert_version, max_len
from config import attention_heads, residual_blocks, decoder_blocks


class Trainer:
    def __init__(self, use_tpu=False):
        """ 
        Distributed Training Object
        """
        if self.use_tpu:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver.connect()
            strategy = tf.distribute.TPUStrategy(tpu)

            encoder = TFBertModel.from_pretrained(bert_version)
            critic = LinearResNet2D(hidden_dim, residual_blocks)

            closeGAN = CloseGAN(
                    embedding_dim=embedding_dim, 
                    latent_dim=hidden_dim,
                    max_length=max_len,
                    transformer_blocks=decoder_blocks,
                    attention_heads=attention_heads,
                    residual_blocks=residual_blocks
                    )
            
        else:
            encoder = TFBertModel.from_pretrained(bert_version)
            critic = LinearResNet2D(hidden_dim, residual_blocks)

            closeGAN = CloseGAN(
                    embedding_dim=embedding_dim, 
                    latent_dim=hidden_dim,
                    transformer_blocks=decoder_blocks,
                    attention_heads=attention_heads,
                    residual_blocks=residual_blocks
                    )
            
    def reconstruction_training(self, dataset, epochs):
        loss_object = keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
        train_loss_results = []

        for epoch in range(epochs):
            epoch_loss_avg = tf.keras.metrics.Mean()

            for x, y in dataset.train:
                loss_value, grads = self.__reconstruction_grad(loss_object, x, y)
                optimizer.apply_gradients(zip(grads, self.encoder.trainable_variables))
                optimizer.apply_gradients(zip(grads, self.decoder.trainable_variables))

                epoch_loss_avg.update_state(loss_value)

        train_loss_results.append(epoch_loss_avg.result())
        return
    

    def adversarial_training(self, dataset, epochs):
        pass
    

    def contrastive_training(self, dataset, epochs):
        pass


    def __reconstruction_grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.__reconstruction_loss(inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)
    

    def __reconstruction_loss(self, loss_object, x, y, training=False):
        z = self.encoder(x[0], training=training)
        y_ = self.decoder([z, x[1]])
        return loss_object(y_true=y, y_pred=y_)


    def __sample_real_z(self, input):
            input_ids, token_type_ids, attention_mask = input
            return self.encoder(
                        input_ids, 
                        token_type_ids=token_type_ids, 
                        attention_mask=attention_mask
                        )

    def cosine_similarity(self, a, b):
        average_a = tf.einsum('ij->j', a) / a.shape[0]
        average_b = tf.einsum('ij->j', b) / b.shape[0]
        return tf.einsum('i,j->', a, b) / (tf.norm(a) * tf.norm(b))