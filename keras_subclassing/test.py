import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split

file_path = r"D:\ruin\data\IMDB Dataset2.csv"
glove_path = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"

def preprocessing_df(file_dir):
    imdb_csv = file_dir
    df_imdb = pd.read_csv(imdb_csv)
    df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

    return df_imdb


def making_parameter():
    df_imdb = preprocessing_df(file_path)

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

    return vocab_size, sequences, maxlen, embedding_matrix


def making_data():
    df_imdb = preprocessing_df(file_path)

    train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
    test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

    x_train, y_train = data_preprocessing(train_df)
    x_test, y_test = data_preprocessing(test_df)
    x_val, y_val = data_preprocessing(val_df)

    return x_train, y_train, x_test, y_test, x_val, y_val


def data_preprocessing(df_data):
    t = Tokenizer()
    vocab_size, sequences, maxlen = making_parameter()

    x_train = df_data['text'].values
    x_train = t.texts_to_sequences(x_train)
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)

    y_train = df_data['label'].values
    y_train = to_categorical(np.asarray(y_train))

    return x_train, y_train

x_train, y_train, x_test, y_test, x_val, y_val = making_data()
