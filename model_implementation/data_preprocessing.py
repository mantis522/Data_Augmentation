import pandas as pd

base_data = pd.read_csv(r"D:\ruin\data\imdb_summarization\t5_base_with_huggingface_sentiment.csv")
base_data = base_data.drop(['Unnamed: 0'], axis=1)


original_text_df = base_data['original_text']
original_label_df = base_data['original_label']
summarized_text_df = base_data['summarized_text']
huggingface_sentiment_df = base_data['huggingface_sentiment']

num_list = []
non_num_list = []

for i in range(len(original_text_df)):
    if original_label_df[i] == huggingface_sentiment_df[i]:
        num_list.append(i)
    else:
        non_num_list.append(i)

train_df = base_data.loc[num_list]
test_df = base_data.loc[non_num_list]

# 훈련 데이터는 4만개, 테스트 데이터는 1만개.
#

cal_per = len(train_df) / 50
cal_per = int(cal_per)

train_df = train_df[:cal_per]
train_origin_df = train_df.drop(['summarized_text', 'huggingface_sentiment'], axis=1)
train_aug_df = train_df.drop(['original_text', 'original_label'], axis=1)

train_aug_df.columns = ['original_text', 'original_label']

train_df = pd.concat([train_origin_df, train_aug_df])

print(len(train_df))