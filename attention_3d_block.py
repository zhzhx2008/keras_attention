#coding=utf-8
from keras import Input, Model
from keras.layers import Dense, Lambda, dot, Activation, concatenate, LSTM, multiply


# https://github.com/philipperemy/keras-attention-mechanism/blob/5d96237ef6/attention.py
def attention_3d_block(hidden_states):
    # @author: felixhao28.
    # hidden_states.shape = (batch_size, time_steps, hidden_size)
    hidden_size = int(hidden_states.shape[2])
    # Inside dense layer
    #              hidden_states            dot               W            =>           score_first_part
    # (batch_size, time_steps, hidden_size) dot (hidden_size, hidden_size) => (batch_size, time_steps, hidden_size)
    # W is the trainable weight matrix of attention Luong's multiplicative style score
    score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
    #            score_first_part           dot        last_hidden_state     => attention_weights
    # (batch_size, time_steps, hidden_size) dot   (batch_size, hidden_size)  => (batch_size, time_steps)
    h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
    score = dot([score_first_part, h_t], [2, 1], name='attention_score')
    attention_weights = Activation('softmax', name='attention_weight')(score)
    # (batch_size, time_steps, hidden_size) dot (batch_size, time_steps) => (batch_size, hidden_size)
    context_vector = dot([hidden_states, attention_weights], [1, 1], name='context_vector')
    pre_activation = concatenate([context_vector, h_t], name='attention_output')
    attention_vector = Dense(hidden_size, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
    return attention_vector


# temporal attention
TIME_STEPS = 72
INPUT_DIM = 300
inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
rnn_out = LSTM(INPUT_DIM, return_sequences=True)(inputs)
attention_output = attention_3d_block(rnn_out)
output = Dense(INPUT_DIM, activation='sigmoid', name='output')(attention_output)
m = Model(inputs=[inputs], outputs=[output])
print(m.summary())


# dense attention
inputs = Input(shape=(INPUT_DIM,))
attention_probs = Dense(INPUT_DIM, activation='softmax', name='attention_probs')(inputs)
attention_mul = multiply([inputs, attention_probs])
m = Model(inputs=[inputs], outputs=[attention_mul])
print(m.summary())
