import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional, LSTM, Dropout, Concatenate, Layer
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = Dense(units)
    self.W2 = Dense(units)
    self.V = Dense(1)

  def call(self, values, query): # 단, key와 value는 같음
    # query shape == (batch_size, hidden size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden size)
    # score 계산을 위해 뒤에서 할 덧셈을 위해서 차원을 변경해줍니다.
    hidden_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(values) + self.W2(hidden_with_time_axis)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights


class BIRNN(tf.keras.Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='sigmoid',
                 ):
        super(BIRNN, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.embedding = Embedding(max_features, embedding_dims, input_length=maxlen, mask_zero=True)
        self.lstm1 = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True))
        self.lstm2 = Bidirectional(LSTM(64, dropout=0.5, return_sequences = True, return_state=True))
        self.dense1 = Dense(20, activation='relu')
        self.dropout = Dropout(0.5)
        self.classifier = Dense(self.class_num, activation=self.last_activation)
        self.attention = BahdanauAttention(64)

    def call(self, inputs):
        emb = self.embedding(inputs)
        net = self.lstm1(emb)
        net, forward_h, forward_c, backward_h, backward_c = self.lstm2(net)
        state_h = Concatenate()([forward_h, backward_h])
        context_vector, _ = self.attention(net, state_h)
        net = self.dense1(context_vector)
        net = self.dropout(net)
        net = self.classifier(net)

        return net

class_num = 1
maxlen = 500
embedding_dims = 128
epochs = 10
batch_size = 256
max_features = 10000

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = max_features)

X_train = pad_sequences(X_train, maxlen=maxlen)
X_test = pad_sequences(X_test, maxlen=maxlen)

model = BIRNN(maxlen=maxlen, max_features=max_features, embedding_dims=embedding_dims
              , class_num=class_num)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

history = model.fit(X_train, y_train, epochs = epochs, batch_size = batch_size,
                    validation_data=(X_test, y_test), verbose=1)
