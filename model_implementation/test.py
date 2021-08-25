import pandas as pd
from transformers import pipeline
import time

start = time.time()

classifier = pipeline('sentiment-analysis')

data = pd.read_csv(r"D:\ruin\data\imdb_summarization\imdb_bart_sum.csv")
data.drop(['Unnamed: 0'], axis=1, inplace=True)

count = 0

original_text_list = []
summarized_text_list = []
original_label_list = []
huggingface_sentiment_list = []

for number in range(len(data)):
    original_text = data['original_text'][number]
    summarized_text = data['t5-large_text'][number]
    original_label = data['original_label'][number]

    huggingface_classifier = classifier(summarized_text)
    huggingface_sentiment = huggingface_classifier[0]['label']

    if huggingface_sentiment == 'NEGATIVE':
        huggingface_sentiment = 0
    elif huggingface_sentiment == 'POSITIVE':
        huggingface_sentiment = 1

    original_text_list.append(original_text)
    original_label_list.append(original_label)
    summarized_text_list.append(summarized_text)
    huggingface_sentiment_list.append(huggingface_sentiment)

    count += 1

    if count % 1000 == 0:
        print(count)

dict_df = {'original_text': original_text_list, 'original_label': original_label_list, 'summarized_text': summarized_text_list,
           'huggingface_sentiment': huggingface_sentiment_list}
dict_df = pd.DataFrame(dict_df)

dict_df.to_csv("bart_sum_with_huggingface_sentiment.csv")

print("time :", time.time() - start)