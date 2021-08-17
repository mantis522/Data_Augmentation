import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Bidirectional, LSTM, Concatenate, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
import os

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

class BahdanauAttention(tf.keras.Model):
    def __init__(self):
        super(BahdanauAttention, self).__init__()
        self.W1 = Dense(64)
        self.W2 = Dense(64)
        self.V = Dense(1)

    def call(self, values, query):
        ## query_shape == (batch_size, hidden_size)
        ## hiden_with_time_axis == (batch_size, 1, hidden_size)
        ## score계산을 위해 (뒤에서 할 덧셈을 위해) 차원 변경.

        hidden_with_time_axis = tf.expand_dims(query, 1)

        ## score shape == (batch_size, max_length, 1)
        ## self.V 때문에 마지막에 1이 됨.
        ## self.V를 하기 직전 shape는 (batch_size, max_length, units)
        score = self.V(tf.nn.tanh(self.W1(values) + self.W2(hidden_with_time_axis)))

        ## attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        ## context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Embedding_layer = Embedding(vocab_size, 128, input_length=max_len, mask_zero=True)
        self.LSTM_layer1 = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True))
        self.LSTM_layer2 = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))
        # self.LSTM_layer2, self.forward_h, self.forward_c, self.backward_h, self.backward_c = Bidirectional(LSTM(64, dropout=0.5, return_sequences=True, return_state=True))
        # self.state_h = Concatenate()
        # self.state_c = Concatenate()
        # self.attention = BahdanauAttention(64)
        self.context_vector, self.attention_weights = self.attention(self.LSTM_layer2, self.state_h)
        self.Dense1 = Dense(20, activation='relu')
        self.Dropout = Dropout(0.5)
        self.Dense2 = Dense(1, activation='sigmoid')

    def call(self, input):
        net = self.Embedding_layer(input)
        net = self.LSTM_layer1(net)
        net, forward_h, forward_c, backward_h, backward_c = self.LSTM_layer2(net)
        state_h = Concatenate()([forward_h, backward_h])
        state_c = Concatenate()([forward_c, backward_c])
        attention = BahdanauAttention(64)
        context_vector, attention_weight = attention(net, state_h)
        net = self.Dense1(net)
        net = (net)(context_vector)
        net = self.Dropout(net)
        net = self.Dense2(net)
        net = Model(inputs=sequence_input, outputs=net)

        return net
#
# sequence_input = Input(shape=(max_len,), dtype='int32')
#
# model = MyModel()
# model = model(sequence_input)
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
#
# model.summary()