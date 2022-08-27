import pandas as pd

test_dir = "....csv"
test_data = pd.read_csv(test_dir)

ori_text_list = []
ori_label_list = []
aug_text_list = []
aug_label_list = []

for i in range(len(test_data)):
    ori_text_list.append(test_data['original_text'][i])
    aug_text_list.append(test_data['summarized_text'][i])
    if test_data['original_label'][i] == 1:
        # test_data['original_label'][i] = 0
        ori_label_list.append(0)
    elif test_data['original_label'][i] == 2:
        # test_data['original_label'][i] = 1
        ori_label_list.append(1)
    elif test_data['huggingface_sentiment'][i] == 1:
        aug_label_list.append(0)
    elif test_data['huggingface_sentiment'][i] == 2:
        aug_label_list.append(1)

df = pd.DataFrame((zip(ori_text_list, ori_label_list, aug_text_list, aug_label_list)), columns=['original_text', 'original_label', 'summarized_text', 'huggingface_sentiment'])
df.to_csv("test.csv", index=False)