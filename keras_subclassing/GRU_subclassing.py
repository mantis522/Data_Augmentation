import numpy as np
from tensorflow.keras.datasets import imdb
import re
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

(X_train, y_train), (X_test, y_test) = imdb.load_data()
num_classes = max(y_train) + 1

len_result = [len(s) for s in X_train]

unique_elements, counts_elements = np.unique(y_train, return_counts=True)

word_to_index = imdb.get_word_index()
index_to_word={}
for key, value in word_to_index.items():
    index_to_word[value+3] = key

for index, token in enumerate(("<pad>", "<sos>", "<unk>")):
  index_to_word[index]=token

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words = vocab_size)

max_len = 500
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

print(y_train.shape)

class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.Embedding_layer = tf.keras.layers.Embedding(vocab_size, 100)
        self.GRU_layer = tf.keras.layers.GRU(128)
        self.Dense_layer = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input):
        net = self.Embedding_layer(input)
        net = self.GRU_layer(net)
        net = self.Dense_layer(net)

        return net

model = MyModel()
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('../model_implementation/GRU_model', monitor='val_acc', mode='max', verbose=1, save_best_only=True, format='tf')
history = model.fit(X_train, y_train, epochs=15,  callbacks=[es, mc], batch_size=60, validation_split=0.2)

loaded_model = load_model('../model_implementation/GRU_model')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))

# model = Sequential()
# model.add(Embedding(vocab_size, 100))
# model.add(GRU(128))
# model.add(Dense(1, activation='sigmoid'))
#
# model.summary()

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
# history = model.fit(X_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_split=0.2)
#
# loaded_model = load_model('GRU_model.h5')
# print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))