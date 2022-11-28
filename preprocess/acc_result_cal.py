import pandas as pd

aug_name = r"C:\Users\RUIN\PycharmProjects\Data_Augmentation\for10_imple_codes\result\Amazon\CNN\T5_Large\Amazon_Aug_CNN_t5_large_glove.csv"
normal_name = r"C:\Users\RUIN\PycharmProjects\Data_Augmentation\for10_imple_codes\result\Amazon\CNN\BART\Amazon_Normal_CNN_bart_glove.csv"
# normal 파일은 분류 모델당 하나만 있기 때문에 딱히 파일 경로 바꿔줄 필요 없음.

result_data = pd.read_csv(aug_name)
normal_data = pd.read_csv(normal_name)

def making_cal_csv(numbers):
    # numbers에는 CNN, GRU의 경우는 10, BERT는 5 넣기.
    start_list = []
    end_list = []
    avg_list = []
    normal_avg_list = []
    for i in range(len(result_data)):
        try:
            if result_data['numbers'][i] == numbers and result_data['numbers'][i + 1] == 1:
                start_list.append(result_data['start'][i])
                end_list.append(result_data['end'][i])
                avg_list.append(result_data['average_acc'][i])
                normal_avg_list.append(normal_data['average_acc'][i])

        except KeyError:
            start_list.append(result_data['start'][len(result_data)-1])
            end_list.append(result_data['end'][len(result_data)-1])
            avg_list.append(result_data['average_acc'][len(result_data)-1])
            normal_avg_list.append(normal_data['average_acc'][len(normal_data)-1])

    df = pd.DataFrame(zip(start_list, end_list, normal_avg_list, avg_list), columns=['start', 'end', 'normal_acc', 'aug_acc'])

    return df

csv_name = r"C:\Users\RUIN\PycharmProjects\Data_Augmentation\for10_imple_codes\result\Amazon\accuracy_csv\Amazon_CNN_t5_large_accuracy.csv"

df_acc = making_cal_csv(10)
df_acc.to_csv(csv_name, index=False)