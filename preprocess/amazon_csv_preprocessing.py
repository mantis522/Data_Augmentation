import pandas as pd
from tqdm import tqdm

# amazon_data = r"D:\ruin\data\amazon\amazon_review_polarity_csv\train.csv"
#
# amazon_data = pd.read_csv(amazon_data)
#
# sentiment_label = amazon_data.iloc[:, 0]
# title_label = amazon_data.iloc[:, 1]
# text_label = amazon_data.iloc[:, 2]
#
# sentiment_list = []
# title_list = []
# text_list = []
#
# for i in range(len(sentiment_label)):
#     sentiment_list.append(sentiment_label[i])
#     title_list.append(title_label[i])
#     text_list.append(text_label[i])
#
# df = pd.DataFrame((zip(sentiment_list, title_list, text_list)), columns = ['sentiment_label', 'title', 'text'])
#
# df.to_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\train_revise.csv", index=False)

amazon_dataset = pd.read_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\train_revise.csv")

pos_label_list = []
neg_label_list = []
pos_title_list = []
neg_title_list = []
pos_text_list = []
neg_text_list = []

for i in range(len(amazon_dataset)):
    if(amazon_dataset['sentiment_label'][i] == 1):
        neg_text = amazon_dataset['text'][i]
        neg_label = amazon_dataset['sentiment_label'][i]
        neg_title = amazon_dataset['title'][i]
        neg_label_list.append(neg_label)
        neg_title_list.append(neg_title)
        neg_text_list.append(neg_text)

    elif(amazon_dataset['sentiment_label'][i] == 2):
        pos_text = amazon_dataset['text'][i]
        pos_label = amazon_dataset['sentiment_label'][i]
        pos_title = amazon_dataset['title'][i]
        pos_label_list.append(pos_label)
        pos_title_list.append(pos_title)
        pos_text_list.append(pos_text)

neg_df = pd.DataFrame((zip(neg_label_list, neg_title_list, neg_text_list)), columns = ['sentiment_label', 'title', 'text'])
neg_df.to_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_neg_train.csv", index=False)

pos_df = pd.DataFrame((zip(pos_label_list, pos_title_list, pos_text_list)), columns = ['sentiment_label', 'title', 'text'])
pos_df.to_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_pos_train.csv", index=False)