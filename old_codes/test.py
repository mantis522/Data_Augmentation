import re
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU, Embedding
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


file_path = r"D:\ruin\data\imdb_summarization\t5_base_with_huggingface_sentiment.csv"
glove_path = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"

imdb_csv = file_path
base_data = pd.read_csv(imdb_csv)
base_data = base_data.drop(['Unnamed: 0'], axis=1)

text_encoding = base_data['original_text']

original_text_df = base_data['original_text']
original_label_df = base_data['original_label']
summarized_text_df = base_data['summarized_text']
huggingface_sentiment_df = base_data['huggingface_sentiment']

num_list = []
non_num_list = []

for i in range(len(original_text_df)):
    if original_label_df[i] == huggingface_sentiment_df[i]:
        num_list.append(i)
    else:
        non_num_list.append(i)

train_df = base_data.loc[num_list]
test_df = base_data.loc[non_num_list]

cal_per = len(train_df)
cal_per = int(cal_per)

train_df = train_df[:cal_per]
# train_df = train_df.drop(['summarized_text', 'huggingface_sentiment'], axis=1)
train_origin_df = train_df.drop(['summarized_text', 'huggingface_sentiment'], axis=1)
train_aug_df = train_df.drop(['original_text', 'original_label'], axis=1)

train_aug_df.columns = ['original_text', 'original_label']

train_df = pd.concat([train_origin_df, train_aug_df])

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


model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(GRU(128))
model.add(Dense(2, activation='softmax'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=1)
mc = ModelCheckpoint('GRU_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(x_train, y_train, epochs=15, callbacks=[es, mc], batch_size=60, validation_data=(x_val, y_val))

loaded_model = load_model('GRU_model.h5')
print("\n 테스트 정확도: %.4f" % (loaded_model.evaluate(x_test, y_test)[1]))