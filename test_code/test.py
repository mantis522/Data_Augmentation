import pandas as pd
import os
import re
import natsort

file_path = r"D:\ruin\data\eda_nlp\imdb_aug"
output_path = r"D:\ruin\data\eda_nlp\imdb_aug_csv"
file_list = os.listdir(file_path)
file_list = natsort.natsorted(file_list)

for input_file in file_list:
    extracted_num = re.sub('.txt', '', input_file)
    output_file = extracted_num + '.csv'

    text_file = file_path + '\\' + input_file

    text_lst = []
    label_lst = []

    f = open(text_file, 'r')
    lines = f.readlines()

    for line in lines:
        label = line.split('\t')[0]
        text = line.split('\t')[1]
        text = re.sub('\n', '', text)
        text_lst.append(text)
        label_lst.append(label)
    f.close()

    df = pd.DataFrame(list(zip(text_lst, label_lst)), columns =['original_text', 'original_label'])

    print(df.to_csv(output_path + '\\' + output_file, index=False))


#
# text_file = r"D:\ruin\data\eda_nlp\imdb_aug\aug500.txt"
#
# text_lst = []
# label_lst = []
#
# f = open(text_file, 'r')
# lines = f.readlines()
#
# for line in lines:
#     label = line.split('\t')[0]
#     text = line.split('\t')[1]
#     text = re.sub('\n', '', text)
#     text_lst.append(text)
#     label_lst.append(label)
# f.close()
#
# df = pd.DataFrame(list(zip(text_lst, label_lst)), columns =['original_text', 'original_label'])
#
# print(df.to_csv('test.csv', index=False))