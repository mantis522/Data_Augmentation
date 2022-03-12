import pandas as pd
import numpy as np
import re

csv_file = r"C:\Users\ruin\PycharmProjects\Data_Augmentation\for10_imple_codes\result\B_Attention\Aug_Attention_biGRU_t5_large.csv"

csv_file = pd.read_csv(csv_file)

CNN_acc = csv_file['acc']

CNN_acc = CNN_acc[1100:]
CNN_acc = np.array(CNN_acc.tolist())

print(CNN_acc)

def clean_text(inputString):
    text_rmv = re.sub('[%]', ' ', inputString)
    text_rmv = text_rmv.replace(" ", "")
    return text_rmv

def cal_acc(accuracy):
    acc_list = []
    for i in range(len(accuracy)):
        present_accuracy = clean_text(accuracy[i])
        present_accuracy = float(present_accuracy)
        acc_list.append(present_accuracy)

    acc_sum = sum(acc_list)
    sum_average = acc_sum / len(acc_list)

    return sum_average

cal = cal_acc(CNN_acc)

print(cal)