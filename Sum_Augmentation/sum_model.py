from transformers import pipeline
import os
import pandas as pd
import time
import torch

start = time.time()

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# using pipeline API for summarization task
summarization = pipeline("summarization", device=0)
imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"

df_imdb = pd.read_csv(imdb_data)

original_text = df_imdb['text'][156]


# summary_text = summarization(original_text, max_length=300, min_length=40)[0]['summary_text']
print("original:", original_text)
# print("Summary:", summary_text)
#
#
# print("time :", time.time() - start)