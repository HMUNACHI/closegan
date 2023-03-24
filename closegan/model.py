import keras
import tensorflow as tf
from tokenizer import Tokenizer
from resnet import LinearResNet2D
from decoder import ConditionalTransformerDecoder

class CloseGAN(keras.Model):
    def __init__(self,
                 embedding_dim, 
                 latent_dim,
                 max_length,
                 transformer_blocks=1,
                 attention_heads=3,
                 residual_blocks=1):
        
        super().__init__()
        self.max_length = max_length
        self.tokenizer = Tokenizer()
        self.vocab_size = len(self.tokenizer.vocab_list)
        
        self.z_generator = LinearResNet2D(latent_dim, residual_blocks)

        self.decoder = ConditionalTransformerDecoder(
                            self.max_length, 
                            self.vocab_size, 
                            embedding_dim, 
                            latent_dim, 
                            transformer_blocks, 
                            attention_heads
                            )
        
    def generate(self, condition, top_k=1):

        if type(condition) != list:
            condition = [condition]

        condition = self.tokenizer.encode(condition)["input_ids"]
        batch_size = tf.shape(condition)[0]
        noise = tf.random.normal(shape=(batch_size, self.max_length, self.embedding_dim))
        z = self.z_generator(noise)

        return self.__decode(z, condition, top_k)


    def __decode(self, z, sequence, tokens_generated, top_k):
        """Top-K sampling """
        tokens_generated = 0
        stop = False

        while stop != True:
            if tokens_generated >= self.max_length:
                stop = True

            # Get probabilities from the last sequence
            output_tokens = self.decoder([z,sequence])
            probs = output_tokens[0, -1, :]
            preds = tf.math.top_k(probs, k=top_k, sorted=True)

            # Get equivalent probabilities for the top_k probabilities
            score = preds[0] / sum(preds[0])

            # Convert indicies to array and probalistically select a choice
            sampled_token_index = tf.random.choice(preds, p=score)
            
            # Check if end of sequence
            if sampled_token_index == 0:
                stop = True
            
            # Decode word and join to sequence
            word = self.tokenizer.decode(sampled_token_index)
            sequence = sequence + " " + word
            tokens_generated += 1

        return  sequence