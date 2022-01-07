from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, GRU, Bidirectional
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
import datetime
import csv

class Attention(Layer):
    def __init__(self, use_bias=True):
        super(Attention, self).__init__()
        self.bias = use_bias
        self.init = initializers.get('glorot_uniform')

    def build(self, input_shape):
        self.ouput_dim = input_shape[-1]
        self.W = self.add_weight(name='{}_W'.format(self.name),
                                 shape=(input_shape[2], 1),
                                 initializer=self.init,
                                 trainable=True)
        if self.bias:
            self.b = self.add_weight(
                name='{}_b'.format(self.name),
                shape=(input_shape[1], 1),
                initializer='zero',
                trainable=True)
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

class TextBiRNNAttention(Model):
    def __init__(self,
                 maxlen,
                 max_features,
                 embedding_dims,
                 class_num,
                 last_activation='softmax',
                 dense_size=None):

        super(TextBiRNNAttention, self).__init__()
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.class_num = class_num
        self.last_activation = last_activation
        self.dense_size = dense_size
        self.embedding = Embedding(input_dim=self.max_features,
                                   output_dim=self.embedding_dims,
                                   input_length=self.maxlen)
        self.bi_rnn = Bidirectional(layer=GRU(
            units=128, activation='tanh', return_sequences=True
        ), merge_mode='concat')
        self.attention = Attention()
        self.classifier = Dense(self.class_num,
                                activation=self.last_activation)

    def call(self, inputs):
        if len(inputs.get_shape()) != 2:
            raise ValueError('The rank of inputs of TextBiRNNAtt must be 2, but now is {}'.format(inputs.get_shape()))
        if inputs.get_shape()[1] != self.maxlen:
            raise ValueError(
                'The maxlen of inputs of TextBiRNNAtt must be %d, but now is %d' % (self.maxlen, inputs.get_shape()[1]))
        emb = self.embedding(inputs)
        x = self.bi_rnn(emb)
        x = self.attention(x)
        if self.dense_size is not None:
            x = self.ffn(x)
        output = self.classifier(x)

        return output

def checkout_dir(dir_path, do_delete=False):
    import shutil
    if do_delete and os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    if not os.path.exists(dir_path):
        print(dir_path, 'make dir ok')
        os.makedirs(dir_path)

class ModelHelper:
    def __init__(self, class_num, maxlen, max_features,
                 embedding_dims, epochs, batch_size):
        self.class_num = class_num
        self.maxlen = maxlen
        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.epochs = epochs
        self.batch_size = batch_size
        self.callback_list = []
        print('Build Model...')
        self.create_model()

    def create_model(self):
        model = TextBiRNNAttention(maxlen=self.maxlen,
                         max_features=self.max_features,
                         embedding_dims=self.embedding_dims,
                         class_num=self.class_num,
                         last_activation='softmax')
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['acc',
                     tf.keras.metrics.Recall(name='recall'),
                     tf.keras.metrics.Precision(name='precision'),
                     tfa.metrics.F1Score(name='F1_micro',
                                         num_classes=2,
                                         average='micro'),
                     tfa.metrics.F1Score(name='F1_macro',
                                         num_classes=2,
                                         average='macro')
                     ],
        )

        # model.summary()
        self.model = model

    def get_callback(self, use_early_stop=True,
                     tensorboard_log_dir='logs\\FastText-epoch-5',
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
    file_path = r"D:\ruin\data\imdb_summarization\t5_large_with_huggingface_sentiment.csv"

    imdb_csv = file_path
    df_imdb = pd.read_csv(imdb_csv)
    df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

    start = 0
    end = 1000

    while start < 50000:
        print("present :", start)

        original_data = df_imdb[start:end]

        # -----------------------------------

        before_concat_origin = np.array(original_data['original_text'].tolist())
        before_concat_origin = list(before_concat_origin)
        before_concat_summ = np.array(original_data['summarized_text'].tolist())
        before_concat_summ = list(before_concat_summ)

        encoding_concat_list = before_concat_summ + before_concat_origin

        text_encoding = encoding_concat_list

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


        # ---------------------------------------
        # ------로 가둬둔 부분은 정수 인코딩을 위해 오리지널 + 요약문 더하고 인코딩한 부분.

        train_df, test_df = train_test_split(original_data, test_size=0.4, random_state=0)
        test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

        train_df = train_df.reset_index(drop=True)
        test_df = test_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)


        def A2D_train(data_df):
            text_list = []
            label_list = []
            for i in range(len(data_df)):
                original_label = int(data_df['original_label'][i])
                huggingface_label = int(data_df['huggingface_sentiment'][i])
                if original_label == huggingface_label:
                    text_list.append(data_df['summarized_text'][i])
                    label_list.append(huggingface_label)

            return text_list, label_list

        sumtext_list, sumlabel_list = A2D_train(train_df)

        def concat_df(data_df, text_list, label_list):
            df = pd.DataFrame([x for x in zip(text_list, label_list)])
            df.columns = ['original_text', 'original_label']
            concating = pd.concat([data_df, df])
            concating = concating.reset_index(drop=True)

            return concating

        train_df = concat_df(train_df, sumtext_list, sumlabel_list)

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

        avg_list = []

        numbers_of_times = 10

        for i in range(numbers_of_times):
            print(i + 1, "번째 학습 시작.")
            use_early_stop = True
            MODEL_NAME = 'Aug_attention'
            tensorboard_log_dir = 'logs\\{}'.format(MODEL_NAME)
            checkpoint_path = 'save_model_dir\\' + MODEL_NAME + '\\cp-{epoch:04d}.ckpt'

            class_num = 2
            embedding_dims = 100
            epochs = 20
            batch_size = 256

            model_helper = ModelHelper(class_num=class_num,
                                       maxlen=maxlen,
                                       max_features=vocab_size,
                                       embedding_dims=embedding_dims,
                                       epochs=epochs,
                                       batch_size=batch_size
                                       )

            model_helper.get_callback(use_early_stop=use_early_stop, tensorboard_log_dir=tensorboard_log_dir,
                                      checkpoint_path=checkpoint_path)
            model_helper.fit(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val)

            print('Restored Model...')
            model_helper = ModelHelper(class_num=class_num,
                                       maxlen=maxlen,
                                       max_features=vocab_size,
                                       embedding_dims=embedding_dims,
                                       epochs=epochs,
                                       batch_size=batch_size
                                       )

            model_helper.load_model(checkpoint_path=checkpoint_path)

            loss, acc, recall, precision, F1_micro, F1_macro = model_helper.model.evaluate(x_test, y_test, verbose=1)

            avg_list.append(float(acc))

            avg_sum = sum(avg_list)
            average_acc = float(avg_sum) / float(i + 1)

            def result_preprocessing(result):
                result = "{:5.2f}%".format(100 * result)
                return result

            loss = result_preprocessing(loss)
            acc = result_preprocessing(acc)
            recall = result_preprocessing(recall)
            precision = result_preprocessing(precision)
            F1_macro = result_preprocessing(F1_macro)
            F1_micro = result_preprocessing(F1_micro)
            average_acc = result_preprocessing(average_acc)

            print("Restored model, accuracy:", acc)
            print("Restored model, recall:", recall)
            print("Restored model, precision:", precision)
            print("Restored model, f1_micro:", F1_micro)
            print("Restored model, f1_macro:", F1_macro)
            print("Average accuracy:", average_acc)

            now = datetime.datetime.now()
            csv_filename = r"C:\Users\ruin\PycharmProjects\Data_Augmentation\for10_imple_codes\result\B_Attention\Aug_Attention_biGRU_t5_large.csv"
            result_list = [now, i + 1, len(original_data), len(train_df), start, end, acc, loss,
                           recall, precision, F1_micro, F1_macro, average_acc]

            if os.path.isfile(csv_filename):
                print("already csv file exist...")

            else:
                print("make new csv file...")
                column_list = ['date', 'numbers', 'number of full data',
                               'number of train data',
                               'start', 'end',
                               'acc', 'loss', 'recall',
                               'precision', 'F1_micro',
                               'F1_macro', 'average_acc']
                df_making = pd.DataFrame(columns=column_list)
                df_making.to_csv(csv_filename, index=False)

            try:
                f = open(csv_filename, 'a', newline='')
                wr = csv.writer(f)
                wr.writerow(result_list)
                f.close()

            except PermissionError:
                print("지금 보고 있는 엑셀창을 닫아주세요.")

            print(i + 1, "번째 학습 끝")

        start = start + 1000
        end = end + 1000

