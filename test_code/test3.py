import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow_hub as hub
import tensorflow_text as tf_text
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ModelCheckpoint
import tensorflow_addons as tfa
import os

file_path = r"D:\ruin\data\IMDB Dataset2.csv"

imdb_csv = file_path
df_imdb = pd.read_csv(imdb_csv)
df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

def df_to_dataset(dataframe, batch_size=128):
    dataframe = dataframe.copy()
    labels = tf.squeeze(tf.constant([dataframe.pop('label')]), axis=0)
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels)).batch(
        batch_size)

    return ds

batch_size = 128

train_ds = df_to_dataset(train_df, batch_size=batch_size)
val_ds = df_to_dataset(val_df, batch_size=batch_size)
test_ds = df_to_dataset(test_df, batch_size=batch_size)

loss = 'categorical_crossentropy'
metrics = ["accuracy"]

def build_classifier_model():
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')

    preprocessing_layer = hub.KerasLayer('https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1',
                                         name='preprocessing')
    encoder_inputs = preprocessing_layer(text_input)

    encoder = hub.KerasLayer('https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/2', trainable=True,
                             name='BERT_encoder')
    outputs = encoder(encoder_inputs)
    net = outputs['pooled_output']
    net = tf.keras.layers.Dropout(0.2)(net)
    net = tf.keras.layers.Dense(1, activation='softmax', name='classifier')(net)
    return tf.keras.Model(text_input, net)

classifier_model = build_classifier_model()
classifier_model.compile(optimizer='adam',
                         loss=loss,
                         metrics=metrics)

history = classifier_model.fit(x=train_ds,
                               validation_data=val_ds,
                               epochs=5)