import pandas as pd

dir_name = "/Users/ruin/Desktop/Data_Augmentation/for10_imple_codes/result/CNN/T5_Base/Aug_CNN_t5_base.csv"

test_data = pd.read_csv(dir_name)


for i in range(len(test_data)):
    try:
        if test_data['end'][i] != test_data['end'][i+1]:
            print(test_data['end'][i])
            print(test_data['average_acc'][i])
    except KeyError:
        print(",,,,")