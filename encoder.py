import tensorflow as tf 

class Encoder(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, units):
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(units,
                                   return_sequences=True,  
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    def call(self,inputs):
        x=self.embedding(inputs)
        output, state =self.gru(x)
        return output, state 
