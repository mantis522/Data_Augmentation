import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import pandas as pd

# 코드는 문제가 없는데 문장의 길이가 길면 OOM이 걸림.
# 그렇다고 배치사이즈를 너무 줄이면 학습하는데 시간도 오래걸리고 성능도 쓰레기됨


tf.random.set_seed(1234)
np.random.seed(1234)

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

print(maxlen)