import json
import pandas as pd
from keras.models import Sequential
import keras
from tensorflow.keras.layers import Dense, LSTM, Embedding, Dropout, Flatten, SpatialDropout1D, Input
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np



origin_neg_directory = "C:/Users/ruin/Desktop/data/json_data/train_neg_full.json"
origin_pos_directory = "C:/Users/ruin/Desktop/data/json_data/train_pos_full.json"
origin_directory = "C:/Users/ruin/Desktop/data/train_data_full.json"
test_directory = "C:/Users/ruin/Desktop/data/json_data/test_data_full.json"

home_origin_dir = "D:/ruin/data/json_data/train_data_full.json"
home_test_dir = "D:/ruin/data/json_data/test_data_full.json"



def making_origin_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    train_review = []
    train_label = []

    train_data = json_data['data']

    for a in range(len(train_data)):
        train_review.append(train_data[a]['txt'])
        train_label.append(train_data[a]['label'])

    df_train = pd.DataFrame(train_review, columns=['data'])
    df_train['label'] = train_label

    return df_train

def making_test_df(file_directory):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    test_data = json_data['data']
    test_review = []
    test_label = []

    for a in range(len(test_data)):
        test_review.append(test_data[a]['txt'])
        test_label.append(test_data[a]['label'])

    df = pd.DataFrame(test_review, columns=['data'])
    df['label'] = test_label

    return df

origin_train_df = making_origin_df(home_origin_dir)
test_df = making_test_df(home_test_dir)

origin_train_df = pd.concat([origin_train_df] * 1, ignore_index=True)


# review_data = origin_train_df['data'].tolist()
# review_label = origin_train_df['label'].tolist()
# test_data = test_df['data'].tolist()
# test_label = test_df['label'].tolist()

X_train = origin_train_df['data'].values
y_train = origin_train_df['label'].values

X_val = test_df['data'].values
y_val = test_df['label'].values

vocab_size = 10000
# max_features = 500

tokenizer = Tokenizer(num_words=vocab_size)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_val_seq = tokenizer.texts_to_sequences(X_val)

l = [len(i) for i in X_train_seq]
l = np.array(l)

print('minimum number of words: {}'.format(l.min()))
print('median number of words: {}'.format(np.median(l)))
print('average number of words: {}'.format(l.mean()))
print('maximum number of words: {}'.format(l.max()))

print(X_train[0])
print(X_train_seq[0])

# # tokenizer = Tokenizer(num_words=max_features, split=' ')
# # tokenizer.fit_on_texts(review_data)
# X_train = tokenizer.texts_to_sequences(review_data)
# X_train = sequence.pad_sequences(X_train, maxlen=max_features)
# Y_train = review_label
#
#
# embed_dim = 128
# lstm_out = 196
#
#
# tokenizer.fit_on_texts(test_data)
# X_test = tokenizer.texts_to_sequences(test_data)
# X_test = sequence.pad_sequences(X_test, maxlen=max_features)
# Y_test = test_label
#
# X_train = np.asarray(X_train).astype('float32')
# Y_train = np.array(Y_train).astype('float32')
# X_test = np.array(X_test).astype('float32')
# Y_test = np.array(Y_test).astype('float32')
#
# # print(X_train.shape, Y_train.shape)
# # print(X_test.shape, Y_train.shape)
#
# # sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#
# model = Sequential()
# model.add(Embedding(max_features, embed_dim, input_length= X_train.shape[1]))
# model.add(SpatialDropout1D(0.4))
# model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
# model.add(Dense(2, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
# model.fit(X_train, Y_train, epochs=7, batch_size=32, verbose=2)


# model = Sequential()
# model.add(Embedding(5000, 120))
# model.add(LSTM(120))
# model.add(Dense(1, activation='sigmoid'))

# es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
# mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)
#
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
# model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=20, batch_size=64, callbacks=[es, mc])