import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow_addons as tfa
import os
import datetime
import csv

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_matrix, text_num):
        super(MyModel, self).__init__()
        # 모델 기술
        self.Embedding_layer = tf.keras.layers.Embedding(input_dim=vocab_size,
                                                         output_dim=100,
                                                         weights=[embedding_matrix],
                                                         input_length=text_num,
                                                         trainable=False)
        self.GRU_layer = tf.keras.layers.GRU(128)
        self.Dense_layer = tf.keras.layers.Dense(2, activation='softmax')
        self.maxlen = text_num

    def call(self, input):
        if len(input.get_shape()) != 2:
            raise ValueError('The rank of inputs of MyModel must be 2, but now is {}'.format(input.get_shape()))
        if input.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of MyModel must be %d, but now is %d' % (self.maxlen, input.get_shape()[1]))

        # 임베딩 레이어에서 input을 넣는 것으로 시작
        # GRU_layer를 거쳐서
        # Dense_layer로 결과 출력.
        # 여기에서 카테고리 분류로 했기 때문에 2, 활성화함수는 소프트맥스
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
        # 모델 생성. 위에서 만든거.
        model = MyModel(vocab_size=self.vocab_size,
                        embedding_matrix=self.embedding_matrix,
                        text_num=self.maxlen)
        # 컴파일. 케라스의 진행은 모델 생성 -> 컴파일로 진행됨.
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

    # early_stop은 사용함.
    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir='logs\\FastText-epoch-5',
                     checkpoint_path="save_model_dir\\cp-moel.ckpt"):

        # callback_list는 콜백에 대한 정보가 들어간다.
        # print 찍어보면 [<keras.callbacks.EarlyStopping object at 0x000001B304626F70>,
        # <keras.callbacks.ModelCheckpoint object at 0x000001B30A55FD60>,
        # <keras.callbacks.TensorBoard object at 0x000001B30A330DC0>]
        # 이런식으로 출력됨.

        callback_list = []
        # 만약 early_stop을 사용한다면 validation_loss값을 대상으로
        # monitor하는 값이 최소가 되어야 하는지, 최대가 되어야 하는지 알려주는 인자.
        # 예를 들어 monitor하는 값이 val_acc 일경우, 값이 클수록 좋기 때문에 'max'
        # val_loss일 경우 작을수록 좋기 때문에 'min'
        # 'auto'는 모델이 알아서 판단한다.
        if use_early_stop:
            early_stopping = EarlyStopping(monitor='val_loss',
                                           patience=3, mode='min')
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
            tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir,
                                               histogram_freq=1)
            callback_list.append(tensorboard_callback)

        self.callback_list = callback_list


    # 모델 fit.
    # train하는 과정으로 훈련 데이터, 검증 데이터 들어감.
    # batch_size, epochs, verbose, callback, validation_data 등이 들어감.

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
    file_path = r"D:\ruin\data\imdb_summarization\t5_large_with_huggingface_sentiment.csv"
    glove_path = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"

    imdb_csv = file_path
    df_imdb = pd.read_csv(imdb_csv)
    df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)
    df_imdb = df_imdb.sample(frac=1).reset_index(drop=True)

    # original_imdb = df_imdb['original_text']
    # sum_imdb = df_imdb['summarized_text']
    numbers = 1000
    original_train_data = df_imdb[:numbers]

    text_encoding = df_imdb['original_text']

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
    epochs = 2

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

    test_df = df_imdb[25000:]
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

    def making_dataset(data_df):
        x_train = data_df['original_text'].values
        x_train = t.texts_to_sequences(x_train)
        x_train = sequence.pad_sequences(x_train, maxlen=maxlen, padding='post')
        y_train = data_df['original_label'].values
        y_train = to_categorical(np.asarray(y_train))

        return x_train, y_train

    x_train, y_train = making_dataset(original_train_data)
    x_test, y_test = making_dataset(test_df)
    x_val, y_val = making_dataset(val_df)

    print('X_train size:', x_train.shape)
    print('y_train size:', y_train.shape)
    print('X_test size:', x_test.shape)
    print('y_test size:', y_test.shape)
    print('X_val size: ', x_val.shape)
    print('y_val size: ', y_val.shape)

    MODEL_NAME = 'TestGRU-epoch-10-emb-100'

    use_early_stop = True
    tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
    checkpoint_path = 'save_model_dir\\' + MODEL_NAME + '\\cp-{epoch:04d}.ckpt'

    # def __init__(self, ...)에 나온 값들을 넣는다.
    model_helper = ModelHelper(batch_size=batch_size, epochs=epochs,
                               vocab_size=vocab_size,
                               embedding_matrix=embedding_matrix,
                               text_num=maxlen)

    # def get_callback에 나온 값들을 넣는다.
    model_helper.get_callback(use_early_stop=use_early_stop,
                              tensorboard_log_dir=tensorboard_log_dir,
                              checkpoint_path=checkpoint_path)

    # def fit에 나온 값들을 넣는다. 본격적인 훈련 부분
    model_helper.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

    # 저장된 모델에 대해 테스트 진행.
    model_helper = ModelHelper(batch_size=batch_size,
                               epochs=epochs, vocab_size=vocab_size,
                               embedding_matrix=embedding_matrix,
                               text_num=maxlen)
    model_helper.load_model(checkpoint_path=checkpoint_path)

    # 위 metric에서 이 평가지표를 전부 넣었기 때문에 여기서 전부 출력.
    loss, acc, recall, precision, F1_micro, F1_macro = model_helper.model.evaluate(x_test, y_test, verbose=1)

    def result_preprocessing(result):
        result = "{:5.2f}%".format(100 * result)
        return result

    loss = result_preprocessing(loss)
    acc = result_preprocessing(acc)
    recall = result_preprocessing(recall)
    precision = result_preprocessing(precision)
    F1_macro = result_preprocessing(F1_macro)
    F1_micro = result_preprocessing(F1_micro)

    print("Restored model, accuracy:", acc)
    print("Restored model, recall:", recall)
    print("Restored model, precision:", precision)
    print("Restored model, f1_micro:", F1_micro)
    print("Restored model, f1_macro:", F1_macro)

    now = datetime.datetime.now()
    csv_filename = r"D:\ruin\data\result\GRU_t5_base_imple.csv"
    result_list = [now, numbers, acc, loss, recall, precision, F1_micro, F1_macro]

    if os.path.isfile(csv_filename):
        print("already csv file exist...")

    else:
        print("make new csv file...")
        column_list = ['date', 'numbers', 'acc', 'loss', 'recall',
                     'precision', 'F1_micro', 'F1_macro']
        df_making = pd.DataFrame(columns=column_list)
        df_making.to_csv(csv_filename, index=False)

    f = open(csv_filename, 'a', newline='')
    wr = csv.writer(f)
    wr.writerow(result_list)
    f.close()