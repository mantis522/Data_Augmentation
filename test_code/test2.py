from transformers import BertTokenizer, TFBertForSequenceClassification
from transformers import InputExample, InputFeatures
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

file_path = r"D:\ruin\data\IMDB Dataset2.csv"

imdb_csv = file_path
df_imdb = pd.read_csv(imdb_csv)
df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

def df_to_dataset(dataframe, batch_size=128):
    labels = tf.squeeze(tf.constant([dataframe.pop('label')]), axis=0)
    ds = tf.data.Dataset.from_tensor_slices((dataframe, labels)).batch(
        batch_size)
    return ds

batch_size = 128
AUTOTUNE = tf.data.AUTOTUNE

train_ds = df_to_dataset(train_df, batch_size=batch_size)
val_ds = df_to_dataset(val_df, batch_size=batch_size)
test_ds = df_to_dataset(test_df, batch_size=batch_size)


train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)