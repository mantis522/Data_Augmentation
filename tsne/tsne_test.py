import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn import preprocessing
import nltk
import re
from tqdm import tqdm
from tensorflow import keras
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
pd.set_option("display.max_rows", 50, "display.max_columns", 50)

# https://www.kaggle.com/code/johanabrahamsson/imdb-sent-analysis-keras-word-emb-t-sne

data = pd.read_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_base_with_huggingface_sentiment.csv")

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