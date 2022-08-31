import pandas as pd
from tqdm import tqdm

large_name = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_large_with_huggingface_sentiment.csv"
base_name = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_base_with_huggingface_sentiment.csv"
bart_name = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_bart_with_huggingface_sentiment.csv"

large_data = pd.read_csv(large_name)
base_data = pd.read_csv(base_name)
bart_data = pd.read_csv(bart_name)

def making_correct_csv(target_data):
    origin_text_list = []
    origin_label_list = []
    sum_text_list = []
    sum_label_list = []

    for i in tqdm(range(len(large_data))):
        large_origin_text = large_data['original_text'][i]

        for j in range(len(target_data)):
            target_origin_text = target_data['original_text'][j]
            target_origin_label = target_data['original_label'][j]
            target_sum_text = target_data['summarized_text'][j]
            target_sum_label = target_data['huggingface_sentiment'][j]

            if large_origin_text == target_origin_text:
                origin_text_list.append(target_origin_text)
                origin_label_list.append(target_origin_label)
                sum_text_list.append(target_sum_text)
                sum_label_list.append(target_sum_label)

    df = pd.DataFrame(zip(origin_text_list, origin_label_list, sum_text_list, sum_label_list),
                      columns=['original_text', 'original_label', 'summarized_text', 'huggingface_sentiment'])

    return df

csv_name = 'test.csv'

df_test = making_correct_csv(bart_data)
df_test.to_csv(csv_name)