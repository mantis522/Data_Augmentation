import pandas as pd
import numpy as np

csv_file = r"D:\ruin\data\imdb_summarization\t5_base_with_huggingface_sentiment.csv"
csv_file = pd.read_csv(csv_file)

csv_file = csv_file.drop(['Unnamed: 0'], axis=1)

start = 0
end = 400
plus = 500

spot = 500

origin_text = np.array(csv_file['original_text'].tolist())
origin_text = list(origin_text)
origin_label = np.array(csv_file['original_label'].tolist())
origin_label = list(origin_label)

print(len(origin_text[0:400]))

while start < 50000:
    train_text = origin_text[start:end]
    train_label = origin_label[start:end]
    test_text = origin_text[end:plus]
    test_label = origin_label[end:plus]

    train_f = open("D:/ruin/data/eda_nlp/imdb/train/" +'train' + str(end) + '.txt', 'w', encoding='utf-8')
    for i in range(len(train_text)):
        data = str(train_label[i]) + '\t' + train_text[i] + '\n'
        train_f.write(data)
    train_f.close()

    test_f = open("D:/ruin/data/eda_nlp/imdb/test/" +'test' + str(plus) + '.txt', 'w', encoding='utf-8')
    for i in range(len(test_text)):
        data = str(test_label[i]) + '\t' + test_text[i] + '\n'
        test_f.write(data)
    test_f.close()

    start = start + spot
    end = end + spot
    plus = plus + spot