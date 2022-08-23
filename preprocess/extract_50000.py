# import pandas as pd
# from collections import Counter
#
# imdb_data = pd.read_csv(r"D:\ruin\data\IMDB Dataset2.csv")
#
# text_count = 0
#
# for i in range(len(imdb_data)):
#     text = imdb_data['text'][i]
#     text_len = len(text)
#     text_count = text_count + text_len
#
# text_avg = text_count / len(imdb_data)
#
# print(text_avg)

import pandas as pd

amazon_dataset = pd.read_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\train_revise.csv")
