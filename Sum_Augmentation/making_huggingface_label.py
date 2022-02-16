import pandas as pd
from transformers import pipeline
import time

# https://discuss.huggingface.co/t/new-pipeline-for-zero-shot-text-classification/681
# https://stackoverflow.com/questions/67849833/how-to-truncate-input-in-the-huggingface-pipeline


start = time.time()

classifier = pipeline('sentiment-analysis', device=0)

class making_label:
    def __init__(self, count):
        self.count = count

    def make(self, dataset, output):

        data = pd.read_csv(dataset)
        data.drop(['Unnamed: 0'], axis=1, inplace=True)

        count = self.count
        original_text_list = []
        summarized_text_list = []
        original_label_list = []
        huggingface_sentiment_list = []

        for number in range(len(data)):
            original_text = data['original_text'][number]
            summarized_text = data['t5-large_text'][number]
            original_label = data['original_label'][number]

            huggingface_classifier = classifier(summarized_text, truncation=True)
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

            if count % 500 == 0:
                print(count)

        dict_df = {'original_text': original_text_list, 'original_label': original_label_list,
                   'summarized_text': summarized_text_list,
                   'huggingface_sentiment': huggingface_sentiment_list}
        dict_df = pd.DataFrame(dict_df)

        dict_df.to_csv(output)

        print("time :", time.time() - start)

t5_base_dir = r"D:\ruin\data\imdb_summarization\imdb_t5_base_sum.csv"
# t5_large_dir = r"D:\ruin\data\imdb_summarization\imdb_t5_large_sum.csv"

t5_base_output = r"D:\ruin\data\imdb_summarization\t5_base_with_huggingface_sentiment.csv"
# t5_large_output = r"D:\ruin\data\imdb_summarization\t5_large_with_huggingface_sentiment.csv"

t5 = making_label(0)
t5.make(t5_base_dir, t5_base_output)
# t5.make(t5_large_dir, t5_large_output)
