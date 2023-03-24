import keras.backend as K
from keras.layers import Layer

class LuongAttentionLayer(Layer):
  def compute_mask(self, inputs, mask=None):
    if mask == None:
      return None
    return mask[1]

  def compute_output_shape(self, input_shape):
    return (input_shape[1][0],input_shape[1][1],input_shape[1][2]*2)


  def call(self, inputs, mask=None):
    encoder_outputs, decoder_outputs = inputs

    # transpose the dimensions of decoder outputs
    decoder_outputs_t = K.permute_dimensions(decoder_outputs, (0,2,1))

    # calculate luong score
    luong_score = K.batch_dot(encoder_outputs,decoder_outputs_t)
    luong_score = K.softmax(luong_score, axis=1) # along the 2nd axis

    # expand the dimensions of luong score and encoded outputs to enable multiplication
    luong_score = K.expand_dims(luong_score, axis=-1) # along last axis
    encoder_outputs = K.expand_dims(encoder_outputs, axis=2) # along 2nd axis

    # get encoded vector
    encoder_vector = encoder_outputs * luong_score
    encoder_vector = K.sum(encoder_vector, axis=1, keepdims=False)

    # [batch,max_dec,2*emb]
    new_decoder_outputs = K.concatenate([decoder_outputs, encoder_vector])

    return new_decoder_outputs