import pandas as pd
import numpy as np

csv_file = r"D:\ruin\data\amazon\amazon_review_polarity_csv\amazon_t5_base_with_huggingface_sentiment.csv"
csv_file = pd.read_csv(csv_file)

csv_file = csv_file.drop(['Unnamed: 0'], axis=1)

start = 0
end = 500

spot = 500

while start < 50000:
    original_data = csv_file[start:end]

    origin_text = np.array(original_data['original_text'].tolist())
    origin_text = list(origin_text)
    origin_label = np.array(original_data['original_label'].tolist())
    origin_label = list(origin_label)

    f = open('D:/ruin/data/eda_nlp/' + 'train' + str(end) + '.txt', 'w', encoding='utf-8')
    for i in range(len(origin_text)):
        data = str(origin_label[i]) + '\t' + origin_text[i] + '\n'
        f.write(data)
    f.close()

    start = start + spot
    end = end + spot