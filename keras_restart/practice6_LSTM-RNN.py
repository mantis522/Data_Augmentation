import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras import Model
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import os
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow_addons as tfa
import pandas as pd
import numpy as np

# 왜인지는 모르겠으나 성능이 폐급 수준임...
# 정확도가 60%는 넘는걸 보면 데이터 전처리 문제는 아닌 것 같은데...
# 그냥 이 모델은 쓰지 말자.

class MyModel(Model):
    def __init__(self, vocab_size, text_num):
        super(MyModel, self).__init__()
        self.Embedding_layer = Embedding(input_dim=vocab_size,
                                                         output_dim=100,
                                                         input_length=text_num,
                                                         trainable=False)
        self.LSTM_layer1 = Bidirectional(LSTM(64, return_sequences=True))
        self.Dropout1 = Dropout(0.3)
        self.LSTM_layer2 = Bidirectional(LSTM(32, return_sequences=False))
        self.Dropout2 = Dropout(0.3)
        self.Dense_layer1 = Dense(20, activation='relu')
        self.Dense_layer2 = Dense(2, kernel_initializer='normal', activation='softmax')
        self.maxlen = text_num

    def call(self, input):
        if len(input.get_shape()) != 2:
            raise ValueError('The rank of inputs of MyModel must be 2, but now is {}'.format(input.get_shape()))
        if input.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of MyModel must be %d, but now is %d' % (self.maxlen, input.get_shape()[1]))

        net = self.Embedding_layer(input)
        net = self.LSTM_layer1(net)
        net = self.Dropout1(net)
        net = self.LSTM_layer2(net)
        net = self.Dropout2(net)
        net = self.Dense_layer1(net)
        net = self.Dense_layer2(net)

        return net

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class ModelHelper:
    def __init__(self, batch_size, epochs, vocab_size, text_num):
        self.batch_size = batch_size
        self.epochs = epochs
        self.vocab_size = vocab_size
        self.maxlen = text_num
        self.callback_list = []
        self.create_model()

    def create_model(self):
        model = MyModel(vocab_size=self.vocab_size,
                        text_num=self.maxlen)
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['acc',
                               tf.keras.metrics.Recall(name='recall'),
                               tf.keras.metrics.Precision(name='precision'),
                               tfa.metrics.F1Score(name='F1_micro',
                                                   num_classes=2,
                                                   average='micro'),
                               tfa.metrics.F1Score(name='F1_macro',
                                                   num_classes=2,
                                                   average='macro')])
        self.model = model

    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):

        callback_list = []
        if use_early_stop:
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=10, mode='min')
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
            checkout_dir(tensorboard_log_dir, do_delete=True)
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir,
                                               histogram_freq=1)
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


if __name__ == '__main__':
    file_path = r"D:\ruin\data\IMDB Dataset2.csv"

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

    train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

    def making_dataset(data_df):
        x_train = data_df['text'].values
        x_train = t.texts_to_sequences(x_train)
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
        x_train = np.array(x_train)
        y_train = list(np.array(data_df['label'].tolist()))
        y_train = to_categorical(np.array(y_train))

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

    MODEL_NAME = 'LSTM-RNN'

    use_early_stop = True
    tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
    checkpoint_path = 'save_model_dir\\' + MODEL_NAME + '\\cp-{epoch:04d}.ckpt'

    model_helper = ModelHelper(batch_size=batch_size, epochs=epochs,
                               vocab_size=vocab_size,
                               text_num=maxlen)

    model_helper.get_callback(use_early_stop=use_early_stop,
                              tensorboard_log_dir=tensorboard_log_dir,
                              checkpoint_path=checkpoint_path)

    model_helper.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    model_helper = ModelHelper(batch_size=batch_size,
                               epochs=epochs, vocab_size=vocab_size,
                               text_num=maxlen)
    model_helper.load_model(checkpoint_path=checkpoint_path)

    loss, acc, recall, precision, F1_micro, F1_macro = model_helper.model.evaluate(x_test, y_test, verbose=1)

    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    print("Restored model, recall : {:5.2f}%".format(100 * recall))
    print("Restored model, precision : {:5.2f}%".format(100 * precision))
    print("Restored model, f1_micro : {:5.2f}%".format(100 * F1_micro))
    print("Restored model, f1_macro : {:5.2f}%".format(100 * F1_macro))