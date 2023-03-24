import tensorflow as tf
from transformers import TFBertTokenizer

class Tokenizer:
    def __init__(self):
        self.tokenizer = TFBertTokenizer.from_pretrained("bert-base-uncased",
                                                         padding="max_length")
        self.vocab_list = self.tokenizer.get_config()["vocab_list"]

    def encode(self, text):
        return self.tokenizer(text)

    def decode(self, encoded_texts):
        decoded_texts = []
        for encoded_text in encoded_texts:
            decoded_texts.append(self.decode_step(encoded_text))
        return decoded_texts

    def decode_step(self, encoded_text):
        decoded_text = []
        for token in encoded_text:
            if token < 1000:
                continue
            current_word = self.vocab_list[token]
            if current_word[:2] == "##":
                previous_word = decoded_text.pop()
                decoded_text.append(previous_word + current_word[2:])
            else:
                decoded_text.append(current_word)
        return " ".join(decoded_text)