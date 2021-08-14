from transformers import pipeline
import pandas as pd
import re
import time

start = time.time()

imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"
df_imdb = pd.read_csv(imdb_data)
df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

text_list = [df_imdb['text'][a] for a in range(len(df_imdb))]
label_list = [df_imdb['label'][a] for a in range(len(df_imdb))]

summarization = pipeline("summarization", device=0)

def cleanText(readData):
    text = re.sub("<[^>]*>", '', readData)
    return text

summarized_list = []

count = 0

for i in range(len(text_list)):
    original_text = text_list[i]

    if len(original_text) > 2056:
        original_text = original_text[:2056]

    summary_text = summarization(original_text, max_length=2056, min_length=40)[0]['summary_text']
    summarized_list.append(summary_text)

    count += 1

    if count % 100 == 0:
        print(count)
        print("time :", time.time() - start)


dict_df = {'original_text': text_list, 'original_label': label_list, 't5-large_text': summarized_list}
dict_df = pd.DataFrame(dict_df)

dict_df.to_csv("imdb_bart_large_sum.csv")

print("time :", time.time() - start)