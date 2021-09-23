import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Bidirectional, Embedding, Dense, GRU
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import os

class Attention(Layer):
    def __init__(self, bias=True):
        super(Attention, self).__init__()
        self.bias = bias
        self.init = tf.keras.initializers.get('glorot_uniform')

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[2], 1),
                                 initializer=self.init,
                                 trainable=True
                                 )
        if self.bias:
            self.b = self.add_weight(
                                     name='{}_b'.format(self.name),
                                     shape=(input_shape[1], 1),
                                     initializer='zero',
                                     trainable=True
                                     )
        else:
            self.b = None

        self.built = True

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs, mask=None):
        score = tf.matmul(inputs, self.W)
        if self.bias:
            score += self.b

        score = tf.tanh(score)
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = inputs * attention_weights
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector

    def get_config(self):
        return {'units': self.output_dim}


class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_matrix, text_num):
        super(MyModel, self).__init__()
        self.maxlen = text_num
        self.Embedding_layer = Embedding(input_dim=vocab_size, output_dim=100,
                                                         weights=[embedding_matrix], input_length=text_num,
                                                         trainable=False)
        self.GRU_layer = Bidirectional(layer=GRU(units=128, activation='tanh', return_sequences=True), merge_mode='concat')
        self.attention = Attention()
        self.Dense_layer = Dense(2, activation='softmax')

    def call(self, input):
        if len(input.get_shape()) != 2:
            raise ValueError('The rank of inputs of MyModel must be 2, but now is {}'.format(input.get_shape()))
        if input.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of MyModel must be %d, but now is %d' % (self.maxlen, input.get_shape()[1]))

        emb = self.Embedding_layer(input)
        net = self.GRU_layer(emb)
        net = self.attention(net)
        net = self.Dense_layer(net)

        return net

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class ModelHelper:
    def __init__(self, batch_size, epochs, vocab_size, embedding_matrix, text_num):
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.embedding_matrix = embedding_matrix
        self.maxlen = text_num
        self.callback_list = []
        self.create_model()

    def create_model(self):
        model = MyModel(vocab_size=self.vocab_size,
                        embedding_matrix=self.embedding_matrix,
                        text_num=self.maxlen)
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['acc'])
        self.model = model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []

        if use_early_stop:
            early_stopping = EarlyStopping(monitor='val_acc', patience=3, mode='max')
            callback_list.append(early_stopping)

        if checkpoint_path is not None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor='val_acc',
                                          mode='max',
                                          save_best_only=True,
                                          save_weights_only=True,
                                          verbose=1,
                                          period=2,
                                          )
            callback_list.append(cp_callback)

        if tensorboard_log_dir is not None:
            # tensorboard --logdir logs/FastText-epoch-5
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1)
            callback_list.append(tensorboard_callback)
        self.callback_list = callback_list

    def fit(self, x_train, y_train, x_val, y_val):
        print('Train...')
        self.model.fit(x_train, y_train,
                       batch_size=self.batch_size,
                       epochs=self.epochs,
                       verbose=1,
                       callbacks=self.callback_list,
                       validation_data=(x_val, y_val))

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname((checkpoint_path))
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        print('restore model name is : ', latest)
        self.model.load_weights(latest)


file_path = r"D:\ruin\data\IMDB Dataset2.csv"
glove_path = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"

imdb_csv = file_path
df_imdb = pd.read_csv(imdb_csv)
df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

text_encoding = df_imdb['text']

t = Tokenizer()
t.fit_on_texts(text_encoding)

vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(text_encoding)

def max_text():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

text_num = max_text()


maxlen = text_num
batch_size = 128
epochs = 15

def Glove_Embedding():
    embeddings_index = {}
    f = open(glove_path, encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    embedding_matrix = np.zeros((vocab_size, 100))

    # fill in matrix
    for word, i in t.word_index.items():  # dictionary
        embedding_vector = embeddings_index.get(word)  # gets embedded vector of word from GloVe
        if embedding_vector is not None:
            # add to matrix
            embedding_matrix[i] = embedding_vector  # each row of matrix

    return embedding_matrix

embedding_matrix = Glove_Embedding()

train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

def making_dataset(data_df):
    x_train = data_df['text'].values
    x_train = t.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    y_train = data_df['label'].values
    y_train = to_categorical(np.asarray(y_train))

    return x_train, y_train

x_train, y_train = making_dataset(train_df)
x_test, y_test = making_dataset(test_df)
x_val, y_val = making_dataset(val_df)

print('X_train size:', x_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', x_test.shape)
print('y_test size:', y_test.shape)
print('X_val size: ', x_val.shape)
print('y_val size: ', y_val.shape)

MODEL_NAME = 'TestGRU-epoch-10-emb-100'

use_early_stop=True
tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
# checkpoint_path = "save_model_dir\\{}\\cp-{epoch:04d}.ckpt".format(MODEL_NAME, '')
checkpoint_path = 'save_model_dir\\'+MODEL_NAME+'\\cp-{epoch:04d}.ckpt'

model_helper = ModelHelper(batch_size=batch_size, epochs=epochs, vocab_size=vocab_size,
                           embedding_matrix=embedding_matrix, text_num=maxlen)

model_helper.get_callback(use_early_stop=use_early_stop,
                          tensorboard_log_dir=tensorboard_log_dir,
                          checkpoint_path=checkpoint_path)

model_helper.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

result = model_helper.model.predict(x_test)
test_score = model_helper.model.evaluate(x_test, y_test,
                            batch_size=batch_size)

print("test loss:", test_score[0], "test accuracy", test_score[1])
print('Restored Model...')

model_helper = ModelHelper(batch_size=batch_size, epochs=epochs, vocab_size=vocab_size,
                           embedding_matrix=embedding_matrix, text_num=maxlen)
model_helper.load_model(checkpoint_path=checkpoint_path)
loss, acc = model_helper.model.evaluate(x_test, y_test, verbose=1)

print("Restored model, accuracy: {:5.2f}%".format(100 * acc))