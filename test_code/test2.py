import os
import natsort
import re

file_path = r"D:\ruin\data\eda_nlp\amazon"
file_list = os.listdir(file_path)
file_list = natsort.natsorted(file_list)

for input_file in file_list:
    extracted_num = re.sub('train', '', input_file)
    extracted_num = re.sub('.txt', '', extracted_num)
    output_file = 'aug' + extracted_num + '.txt'

    print(file_path + '\\' + output_file)

