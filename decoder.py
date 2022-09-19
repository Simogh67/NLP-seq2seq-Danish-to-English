import tensorflow as tf
from attention import Attention

class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                   return_sequences=True,  
                                   return_state=True, 
                                   recurrent_initializer='glorot_uniform')
        self.attention=Attention(units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        
    def call(self,inputs,hidden,encoder_outputs):
        x=self.embedding(inputs)
        context_vector,weights=self.attention(hidden,encoder_outputs)
        context_vector=tf.expand_dims(context_vector, 1)
        x = tf.concat([context_vector, x], axis = -1)
        output, state =self.gru(x,initial_state=hidden)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.fc(output)
        return x, state