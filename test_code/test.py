import pandas as pd
import nltk

file_path = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_large_with_huggingface_sentiment.csv"
df_imdb = pd.read_csv(file_path)


count = 0

for i in range(len(df_imdb)):
    count = count + len(df_imdb['summarized_text'][i].split())

print(count / len(df_imdb))