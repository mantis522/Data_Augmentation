import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
import tensorflow_addons as tfa
import os

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_matrix, text_num):
        super(MyModel, self).__init__()
        self.Embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=100,
                                                         weights=[embedding_matrix], input_length=text_num,
                                                         trainable=False)
        self.GRU_layer = tf.keras.layers.GRU(128)
        self.Dense_layer = tf.keras.layers.Dense(2, activation='softmax')
        self.maxlen = text_num

    def call(self, input):
        if len(input.get_shape()) != 2:
            raise ValueError('The rank of inputs of MyModel must be 2, but now is {}'.format(input.get_shape()))
        if input.get_shape()[1] != self.maxlen:
            raise ValueError('The maxlen of inputs of MyModel must be %d, but now is %d' % (self.maxlen, input.get_shape()[1]))

        net = self.Embedding_layer(input)
        net = self.GRU_layer(net)
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
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc',
                                                                                  tf.keras.metrics.Recall(name='recall'),
                                                                                  tf.keras.metrics.Precision(name='precision'),
                                                                                  tfa.metrics.F1Score(name='F1_micro',
                                                                                                      num_classes=2,
                                                                                                      average='micro'),
                                                                                  tfa.metrics.F1Score(name='F1_macro',
                                                                                                      num_classes=2,
                                                                                                      average='macro')
                                                                                  ])
        self.model = model

    def get_callback(self, use_early_stop=True, tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):
        callback_list = []

        if use_early_stop:
            early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
            callback_list.append(early_stopping)

        if checkpoint_path is not None:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            checkout_dir(checkpoint_dir, do_delete=True)
            cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                          monitor='val_loss',
                                          mode='min',
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


glove_path = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"
base_data = pd.read_csv(r"D:\ruin\data\imdb_summarization\t5_base_with_huggingface_sentiment.csv")
base_data = base_data.drop(['Unnamed: 0'], axis=1)



text_encoding = base_data['original_text']

# original_text_df = base_data['original_text']
# original_label_df = base_data['original_label']
# summarized_text_df = base_data['summarized_text']
# huggingface_sentiment_df = base_data['huggingface_sentiment']
#
# num_list = []
# non_num_list = []
#
# for i in range(len(original_text_df)):
#     if original_label_df[i] == huggingface_sentiment_df[i]:
#         num_list.append(i)
#     else:
#         non_num_list.append(i)
#
# train_df = base_data.loc[num_list]
# test_df = base_data.loc[non_num_list]
#
# # 훈련 데이터는 4만개, 테스트 데이터는 1만개.
# #
#
# cal_per = len(train_df)
# cal_per = int(cal_per)

# train_df = train_df[:cal_per]
# train_origin_df = train_df.drop(['summarized_text', 'huggingface_sentiment'], axis=1)
# train_aug_df = train_df.drop(['original_text', 'original_label'], axis=1)
#
# train_aug_df.columns = ['original_text', 'original_label']
#
# train_df = pd.concat([train_origin_df, train_aug_df])

# text_encoding = train_df['original_text']


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

train_df, test_df = train_test_split(base_data, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

def making_dataset(data_df):
    x_train = data_df['original_text'].values
    x_train = t.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
    y_train = data_df['original_label'].values
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
loss, acc, recall, precision, F1_micro, F1_macro = model_helper.model.evaluate(x_test, y_test, verbose=1)

print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
print("Restored model, recall : {:5.2f}%".format(100 * recall))
print("Restored model, precision : {:5.2f}%".format(100 * precision))
print("Restored model, f1_micro : {:5.2f}%".format(100 * F1_micro))
print("Restored model, f1_macro : {:5.2f}%".format(100 * F1_macro))