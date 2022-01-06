from tensorflow.keras import initializers
from tensorflow.keras.layers import Layer
import tensorflow as tf
from tensorflow.keras.layers import Embedding, Dense, Conv1D, GlobalMaxPool1D
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

