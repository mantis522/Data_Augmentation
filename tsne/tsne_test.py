import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import tensorflow_addons as tfa
import nltk
import re
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
pd.set_option("display.max_rows", 50, "display.max_columns", 50)

# https://www.kaggle.com/code/johanabrahamsson/imdb-sent-analysis-keras-word-emb-t-sne

data = pd.read_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_base_with_huggingface_sentiment.csv")
glove_path = r"D:\ruin\data\glove.6B\glove.6B.100d.txt"

stopWords = nltk.corpus.stopwords.words('english')
snoStemmer = nltk.stem.SnowballStemmer('english')
wnlemmatizer = nltk.stem.WordNetLemmatizer()

def clean_html(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', sentence)
    return cleantext

def clean_punc(word):
    cleaned = re.sub(r'[?|!|\'|#]', r'', word)
    cleaned = re.sub(r'[.|,|)|(|\|/]', r' ', cleaned)
    return cleaned

def filtered_sents(data_frame):
    # Creating a list of filtered sentences:
    final_string = []
    s = ''
    for sentence in tqdm(data_frame):
        filtered_sentence = []
        sentence = clean_html(sentence)
        for word in sentence.split():
            for cleaned_word in clean_punc(word).split():
                if (cleaned_word.isalpha() and (len(cleaned_word) > 2) and cleaned_word not in stopWords):
                    lemmatized_word = wnlemmatizer.lemmatize(cleaned_word.lower())
                    stemmed_word = snoStemmer.stem(lemmatized_word)
                    filtered_sentence.append(stemmed_word)
                else:
                    continue
        strl = ' '.join(filtered_sentence)
        final_string.append(strl)
    return final_string

data.cleaned_review = filtered_sents(data.original_text)

print(data.cleaned_review[0])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(data.cleaned_review)
list_tokenized_data = tokenizer.texts_to_sequences(data.cleaned_review)
word_index = tokenizer.word_index
index_word = dict([(value, key) for (key, value) in word_index.items()])


def max_text():
    for i in range(1, len(list_tokenized_data)):
        max_length = len(list_tokenized_data[0])
        if len(list_tokenized_data[i]) > max_length:
            max_length = len(list_tokenized_data[i])
    return max_length

text_num = max_text()
maxlen = text_num

print(pd.Series(word_index).head())
print('\n')
print(pd.Series(word_index).tail())

length_list = []
for i in list_tokenized_data:
    length_list.append(len(i))

f, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False)
pd.Series(length_list).hist(bins=100, ax = axes[0])
pd.Series(length_list).hist(bins=100, ax = axes[1])
plt.xlim(0,400)
plt.show()


X = tf.keras.preprocessing.sequence.pad_sequences(list_tokenized_data,
                                                        padding='post',
                                                        maxlen=maxlen)

y = data.original_label

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.4, random_state = 0)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size = 0.5, random_state = 0)
print(f'SHAPE: \n X_train: {X_train.shape}, y_train: {y_train.shape}, X_val: {X_val.shape}, y_val:{y_val.shape}, X_test: {X_test.shape}, y_test: {y_test.shape}')

vocab_size = max(np.max(X_train), np.max(X_test), np.max(X_val)) + 1

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
    for word, i in tokenizer.word_index.items():  # dictionary
        embedding_vector = embeddings_index.get(word)  # gets embedded vector of word from GloVe
        if embedding_vector is not None:
            # add to matrix
            embedding_matrix[i] = embedding_vector  # each row of matrix

    return embedding_matrix

embedding_matrix = Glove_Embedding()

model = tf.keras.Sequential()

model.add(tf.keras.layers.Embedding(input_dim=vocab_size,
                                    output_dim=100,
                                    weights=[embedding_matrix],
                                    input_length=maxlen,
                                    trainable=False))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Conv1D(256, 3, padding='valid', activation='relu'))
model.add(tf.keras.layers.GlobalMaxPool1D())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.3))
model.add(tf.keras.layers.Dense(2, activation='softmax'))

model.summary()

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


LEARNING_RATE = 1e-3
OPTIMIZER = tf.keras.optimizers.Adam(lr=LEARNING_RATE)
model.compile(optimizer=OPTIMIZER,
              loss='binary_crossentropy',
              metrics=['acc'])

batch_size = 256
epochs = 20

CALLBACKS = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)]
history = model.fit(X_train, y_train,
                       batch_size=batch_size,
                       epochs=epochs,
                       verbose=1,
                       callbacks=CALLBACKS,
                       validation_data=(X_val, y_val))

loss, acc, recall, precision, F1_micro, F1_macro = model.evaluate(X_test, y_test, verbose=1)

print(acc)