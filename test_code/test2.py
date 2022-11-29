import os
import natsort

file_path = r"D:\ruin\data\eda_nlp\imdb\train_aug_csv"

file_list = os.listdir(file_path)
file_list = natsort.natsorted(file_list)

print(file_list)

for i in range(len(file_list)):
    file_name = file_path + '\\' + file_list[i]
    print(file_name)
