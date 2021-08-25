import pandas as pd
from transformers import pipeline
from nltk.sentiment.vader import SentimentIntensityAnalyzer

analyser = SentimentIntensityAnalyzer()
classifier = pipeline('sentiment-analysis')

data = pd.read_csv(r"D:\ruin\data\imdb_summarization\imdb_bart_sum.csv")
data.drop(['Unnamed: 0'], axis=1, inplace=True)

def vader_polarity(text):
    """ Transform the output to a binary 0/1 result """
    score = analyser.polarity_scores(text)
    return score

number = 300

original_sent = data['original_text'][number]
test_sent = data['t5-large_text'][number]
original_label = data['original_label'][number]


huggingface_sent = classifier(test_sent)
vader_sent = vader_polarity(test_sent)

print(huggingface_sent[0]['label'])
print(vader_sent)
print("original_sentence : ", original_sent)
print("summarized_sentence : ", test_sent)
print(original_label)