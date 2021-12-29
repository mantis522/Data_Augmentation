from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import re

file_path = r"D:\ruin\data\IMDB Dataset2.csv"

imdb_df = pd.read_csv(file_path)
df_imdb = imdb_df.drop(['Unnamed: 0'], axis=1)

train_df, test_df = train_test_split(df_imdb, test_size=0.2, random_state=0)
test_df, val_df = train_test_split(test_df, test_size=0.5, random_state=0)

y = test_df["label"]

y = np.array(list(y))

TAG_RE = re.compile(r"<[^>]+>")

def remove_tags(text):
    return TAG_RE.sub("", text)

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Removing multiple spaces
    sentence = re.sub(r"\s+", " ", sentence)

    return sentence

