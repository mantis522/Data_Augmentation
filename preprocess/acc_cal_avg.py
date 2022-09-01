import pandas as pd
import re
import os

def removing_percent(text):
    text = re.sub('%', '', text)
    text = float(text)

    return text

def making_cal(dir, start, end, acc_type):
    target_data = pd.read_csv(dir)
    avg_count = 0
    count = 0

    for i in range(start, end):
        current_acc = target_data[acc_type][i]
        current_acc = removing_percent(current_acc)
        avg_count = avg_count + current_acc
        count += 1

    avg_acc = round((avg_count / count), 2)

    return avg_acc

# 500은 range 0, 100
# 1000은 100, 150
# 2000은 150, 175
# 5000은 175, 185
# 10000은 185, 190
# 12500은 190, 194
# 25000은 194, 196
# 50000은 196, 197

def avg_print(dir, acc_type):
    acc_500 = making_cal(dir, 0, 100, acc_type)
    acc_1000 = making_cal(dir, 100, 150, acc_type)
    acc_2000 = making_cal(dir, 150, 175, acc_type)
    acc_5000 = making_cal(dir, 175, 185, acc_type)
    acc_10000 = making_cal(dir, 185, 190, acc_type)
    acc_12500 = making_cal(dir, 190, 194, acc_type)
    acc_25000 = making_cal(dir, 194, 196, acc_type)
    acc_50000 = making_cal(dir, 196, 197, acc_type)

    acc_list = [acc_500, acc_1000, acc_2000, acc_5000, acc_10000, acc_12500, acc_25000, acc_50000]

    return acc_list

file_name = r"C:\Users\ruin\PycharmProjects\Data_Augmentation\for10_imple_codes\result\accuracy_csv"
file_list = os.listdir(file_name)

def preprocessing_file_dir(text):
    text = re.sub('.csv', '', text)
    text = re.sub('.DS_Store', '', text)
    text = re.sub('_accuracy', '', text)

    return text

def making_print_cal(acc_type):
    for file in file_list:
        file_dir = file_name + '/' + file
        try:
            print(preprocessing_file_dir(file) + '_' + acc_type)
            print(avg_print(file_dir, acc_type))

        except UnicodeDecodeError:
            pass

print(making_print_cal('normal_acc'))
print(making_print_cal('aug_acc'))