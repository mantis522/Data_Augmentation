from tensorflow.keras import datasets
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
import tensorflow as tf

vocab_size = 10000
(X_train, y_train), (X_test, y_test) = datasets.imdb.load_data(num_words=vocab_size)

max_len = 200
X_train = pad_sequences(X_train, maxlen=max_len)
X_test = pad_sequences(X_test, maxlen=max_len)

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size):
        super(MyModel, self).__init__()
        self.Embedding_layer = tf.keras.layers.Embedding(vocab_size, 256)
        self.Dropout1 = tf.keras.layers.Dropout(0.3)
        self.Conv1D = tf.keras.layers.Conv1D(256, 3, padding='valid', activation='relu')
        self.Maxpooling = tf.keras.layers.GlobalMaxPool1D()
        self.Dense_layer1 = tf.keras.layers.Dense(128, activation='relu')
        self.Dropout2 = tf.keras.layers.Dropout(0.5)
        self.Dense_layer2 = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, input):
        net = self.Embedding_layer(input)
        net = self.Dropout1(net)
        net = self.Conv1D(net)
        net = self.Maxpooling(net)
        net = self.Dense_layer1(net)
        net = self.Dropout2(net)
        net = self.Dense_layer2(net)

        return net

model = MyModel(vocab_size)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('../model_implementation/CNN_model', monitor='val_acc', mode='max', verbose=1, save_best_only=True, format='tf')

history = model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test), callbacks=[es, mc])

loaded_model = load_model('../model_implementation/CNN_model')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(X_test, y_test)[1]))