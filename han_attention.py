#coding=utf-8

import keras.backend as K
from keras import initializers, Input, Model
from keras.engine import Layer
from keras.layers import Embedding, Bidirectional, GRU, TimeDistributed, Dense


class Han_Attention(Layer):
    def __init__(self, **kwargs):
        self.init = initializers.get('normal')
        super(Han_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape)==3
        self.W = K.variable(self.init((input_shape[-1],)))
        self.trainable_weights = [self.W]
        super(Han_Attention, self).build(input_shape)  # be sure you call this somewhere!

    def call(self, x, mask=None):
        eij = K.tanh(K.squeeze(K.dot(x, K.expand_dims(self.W)), axis=-1))
        ai = K.exp(eij)
        weights = ai/K.expand_dims(K.sum(ai, axis=1),1)
        weighted_input = x*K.expand_dims(weights,2)
        res = K.sum(weighted_input, axis=1)
        return res
    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[-1])



vocab_size = 6000
embedding_dim = 300
max_sen_len = 72
max_sens = 32

embedding_layer = Embedding(vocab_size, embedding_dim)
sentence_input = Input(shape=(max_sen_len,), dtype='int32')
embedded_sequences = embedding_layer(sentence_input)
l_lstm = Bidirectional(GRU(100, return_sequences=True))(embedded_sequences)
l_dense = TimeDistributed(Dense(200))(l_lstm)
l_att = Han_Attention()(l_dense)
sentEncoder = Model(sentence_input, l_att)
print(sentEncoder.summary())

review_input = Input(shape=(max_sens, max_sen_len,), dtype='int32')
review_encoder = TimeDistributed(sentEncoder)(review_input)
l_lstm_sent = Bidirectional(GRU(100, return_sequences=True))(review_encoder)
l_dense_sent = TimeDistributed(Dense(200))(l_lstm_sent)
l_att_sent = Han_Attention()(l_dense_sent)
preds = Dense(2, activation='softmax')(l_att_sent)
model = Model(review_input, preds)
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print(model.summary())