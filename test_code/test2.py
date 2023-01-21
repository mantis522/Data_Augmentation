import pandas as pd

file_path = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_large_with_huggingface_sentiment.csv"
df_imdb = pd.read_csv(file_path)

count_pos = 0
count_neg = 0

for i in range(len(df_imdb)):
    if df_imdb['original_label'][i] == 1 and df_imdb['huggingface_sentiment'][i] == 1:
        count_pos = count_pos + 1
    elif df_imdb['original_label'][i] == 0 and df_imdb['huggingface_sentiment'][i] == 0:
        count_neg = count_neg + 1

print(count_pos)
print(count_neg)