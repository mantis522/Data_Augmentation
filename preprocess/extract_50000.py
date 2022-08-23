import pandas as pd
from tqdm import tqdm

neg_data_name = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_neg_train.csv"
pos_data_name = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_pos_train.csv"

def making_df(df_name):
    label_list = []
    title_list = []
    text_list = []
    amazon_data = pd.read_csv(df_name)

    for i in tqdm(range(len(amazon_data))):
        if len(amazon_data['text'][i]) >= 900:
            text_list.append(amazon_data['text'][i])
            label_list.append(amazon_data['sentiment_label'][i])
            title_list.append(amazon_data['title'][i])

    df = pd.DataFrame((zip(label_list, title_list, text_list)), columns=['sentiment_label', 'title', 'text'])

    return df

amazon_data_neg = making_df(neg_data_name)
amazon_data_pos = making_df(pos_data_name)

amazon_data_neg = amazon_data_neg[:25000]
amazon_data_pos = amazon_data_pos[:25000]

amazon_data = pd.concat([amazon_data_neg, amazon_data_pos])
amazon_data.to_csv(r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_50k_data.csv", index=False)