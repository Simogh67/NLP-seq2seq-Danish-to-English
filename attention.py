import tensorflow as tf

class Attention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)
        self.V = tf.keras.layers.Dense(1)
    
    def call(self, query, value):
        query= tf.expand_dims(query, 1)
        w1_query = self.W1(query)
        w2_key = self.W2(value)
        score = self.V(tf.nn.tanh(w2_key + w1_query))
        weights = tf.nn.softmax(score, axis=1)

        context_vector = weights * value
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector,weights