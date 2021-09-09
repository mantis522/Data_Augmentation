import pandas as pd

base_data = pd.read_csv(r"D:\ruin\data\imdb_summarization\t5_base_with_huggingface_sentiment.csv")
base_data = base_data.drop(['Unnamed: 0'], axis=1)


original_text_df = base_data['original_text']
original_label_df = base_data['original_label']
summarized_text_df = base_data['summarized_text']
huggingface_sentiment_df = base_data['huggingface_sentiment']

