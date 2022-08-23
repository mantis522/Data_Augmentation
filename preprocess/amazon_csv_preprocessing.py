import pandas as pd
from tqdm import tqdm

amazon_data = r"D:\ruin\data\amazon\amazon_review_polarity_csv\train.csv"

amazon_data = pd.read_csv(amazon_data)

sentiment_label = amazon_data.iloc[:, 0]
title_label = amazon_data.iloc[:, 1]
text_label = amazon_data.iloc[:, 2]

sentiment_list = []
title_list = []
text_list = []

for i in range(len(sentiment_label)):
    sentiment_list.append(sentiment_label[i])
    title_list.append(title_label[i])
    text_list.append(text_label[i])

df = pd.DataFrame((zip(sentiment_list, title_list, text_list)), columns = ['sentiment_label', 'title', 'text'])

df.to_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\train_revise.csv", index=False)
