from transformers import T5ForConditionalGeneration, T5Tokenizer
import pandas as pd
import re
import time
import torch

start = time.time()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

imdb_data = r"D:\ruin\data\IMDB Dataset2.csv"
df_imdb = pd.read_csv(imdb_data)
df_imdb = df_imdb.drop(['Unnamed: 0'], axis=1)

text_list = [df_imdb['text'][a] for a in range(len(df_imdb))]
label_list = [df_imdb['label'][a] for a in range(len(df_imdb))]

model = T5ForConditionalGeneration.from_pretrained("t5-large")
model = model.to(device)
tokenizer = T5Tokenizer.from_pretrained("t5-large")

def cleanText(readData):
    text = re.sub("<[^>]*>", '', readData)
    return text

summarized_list = []

count = 0

for a in range(len(text_list)):
    inputs = tokenizer.encode(text_list[a], return_tensors="pt", max_length=1024, truncation=True)
    inputs = inputs.to(device)
    outputs = model.generate(
        inputs,
        max_length=300,
        min_length=40,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True)

    summarized_text = tokenizer.decode(outputs[0])
    summarized_list.append(cleanText(summarized_text))
    count += 1

    if count % 100 == 0:
        print(count)

dict_df = {'original_text': text_list, 'original_label': label_list, 't5-large_text': summarized_list}
dict_df = pd.DataFrame(dict_df)

dict_df.to_csv("imdb_t5_large_sum.csv")

print("time :", time.time() - start)