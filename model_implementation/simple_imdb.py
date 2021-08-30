import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import sequence

train_dir = "D:/data/csv_file/amazon_len_renew/amazon_5000_renew.csv"
test_dir = "D:/data/csv_file/amazon_len_renew/test_renew.csv"
glove_100_dir = "D:/data/glove.6B/glove.6B.100d.txt"

removed_amod_neg = "D:/data/json_data/removed_data/amazon/5000/removed_amod_neg_5000.json"
removed_amod_pos = "D:/data/json_data/removed_data/amazon/5000/removed_amod_pos_5000.json"
removed_PP_neg = "D:/data/json_data/removed_data/amazon/5000/removed_PP_neg_5000.json"
removed_PP_pos = "D:/data/json_data/removed_data/amazon/5000/removed_PP_pos_5000.json"
removed_SBAR_neg = "D:/data/json_data/removed_data/amazon/5000/removed_SBAR_neg_5000.json"
removed_SBAR_pos = "D:/data/json_data/removed_data/amazon/5000/removed_SBAR_pos_5000.json"

# home_train_dir = r"D:\ruin\data\csv_file\amazon_len_renew\amazon_100000_renew.csv"
# home_test_dir = r"D:\ruin\data\csv_file\amazon_len_renew\amazon_test.csv"
# home_glove = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"
#
#
# home_removed_amod_neg = r"D:\ruin\data\json_data\removed_data\amazon\100000\removed_amod_neg_100000.json"
# home_removed_amod_pos = r"D:\ruin\data\json_data\removed_data\amazon\100000\removed_amod_pos_100000.json"
# home_removed_PP_neg = r"D:\ruin\data\json_data\removed_data\amazon\100000\removed_PP_neg_100000.json"
# home_removed_PP_pos = r"D:\ruin\data\json_data\removed_data\amazon\100000\removed_PP_pos_100000.json"
# home_removed_SBAR_neg = r"D:\ruin\data\json_data\removed_data\amazon\100000\removed_SBAR_neg_100000.json"
# home_removed_SBAR_pos = r"D:\ruin\data\json_data\removed_data\amazon\100000\removed_SBAR_pos_100000.json"

def making_df(file_directory, label):
    with open(file_directory) as json_file:
        json_data = json.load(json_file)

    removed_sentence = json_data['removed_sentence']
    removed_sentence = removed_sentence[0]
    removed = []

    for a in range(len(removed_sentence)):
        for b in range(len(removed_sentence[a])):
            removed.append(removed_sentence[a][b])

    df = pd.DataFrame(removed)
    df.columns = ['review']
    df['label'] = label

    return df

removed_amod_pos = making_df(removed_amod_pos, 1)
removed_amod_neg = making_df(removed_amod_neg, 0)
removed_PP_pos = making_df(removed_PP_pos, 1)
removed_PP_neg = making_df(removed_PP_neg, 0)
removed_SBAR_neg = making_df(removed_SBAR_neg, 0)
removed_SBAR_pos = making_df(removed_SBAR_pos, 1)

original_train_df = pd.read_csv(train_dir)
original_test_df = pd.read_csv(test_dir)

original_test_df, original_val_df = train_test_split(original_test_df, test_size=0.4, random_state=0)

x = original_train_df['review']
y = original_train_df['label']

# concat_train_df = pd.concat([original_train_df, removed_PP_neg, removed_PP_pos, removed_SBAR_pos, removed_SBAR_neg, removed_amod_pos, removed_amod_neg]).reset_index(drop=True)
concat_train_df = pd.concat([original_train_df, removed_amod_neg, removed_amod_pos, removed_PP_neg, removed_PP_pos]).reset_index(drop=True)

t = Tokenizer()
t.fit_on_texts(x)

vocab_size = len(t.word_index) + 1
sequences = t.texts_to_sequences(x)

def max_text():
    for i in range(1, len(sequences)):
        max_length = len(sequences[0])
        if len(sequences[i]) > max_length:
            max_length = len(sequences[i])
    return max_length

text_num = max_text()
maxlen = text_num

# padded_X = pad_sequences(sequences, padding='post', maxlen=maxlen)

x_train = concat_train_df['review'].values
x_train = t.texts_to_sequences(x_train)
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
y_train = concat_train_df['label'].values
y_train = to_categorical(np.asarray(y_train))

x_test = original_test_df['review'].values
x_test = t.texts_to_sequences(x_test)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
y_test = original_test_df['label'].values
y_test = to_categorical(np.asarray(y_test))

x_val = original_val_df['review'].values
x_val = t.texts_to_sequences(x_val)
x_val = sequence.pad_sequences(x_val, maxlen=maxlen)
y_val = original_val_df['label'].values
y_val = to_categorical(np.asarray(y_val))

print('X_train size:', x_train.shape)
print('y_train size:', y_train.shape)
print('X_test size:', x_test.shape)
print('y_test size:', y_test.shape)
print('X_val size: ', x_val.shape)
print('y_val size: ', y_val.shape)

embeddings_index = {}
f = open(glove_100_dir, encoding='utf-8')
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

embedding_matrix = np.zeros((vocab_size, 100))

# fill in matrix
for word, i in t.word_index.items():  # dictionary
    embedding_vector = embeddings_index.get(word) # gets embedded vector of word from GloVe
    if embedding_vector is not None:
        # add to matrix
        embedding_matrix[i] = embedding_vector # each row of matrix

embedding_layer = Embedding(input_dim=vocab_size, output_dim=100, weights=[embedding_matrix],
                           input_length = text_num, trainable=False)

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=4)
mc = ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', verbose=1, save_best_only=True)

model = Sequential()
model.add(embedding_layer)
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

hist = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=100, batch_size=64, verbose=1, callbacks=[es, mc])

print("Accuracy...")
loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
print("Testing Accuracy:  {:.4f}".format(accuracy))