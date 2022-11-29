import os
import natsort
import pandas as pd

aug_path = r"D:\ruin\data\eda_nlp\imdb\train_aug_csv"
test_path = r"D:\ruin\data\eda_nlp\imdb\test_csv"

batch_size = 256
epochs = 20

start = 0
end = 500

point = 500

aug_list = os.listdir(aug_path)
aug_list = natsort.natsorted(aug_list)
test_list = os.listdir(test_path)
test_list = natsort.natsorted(test_list)

print(aug_list[:10])
print(test_list[:10])

# 원리는 1부터 100까지 더하는 반복문과 동일.
# 비어있는 데이터프레임 만들고 반복하면서 데이터프레임에 데이터 추가하면 됨.

empty_df = pd.DataFrame()

for i in range(10):
    aug_name = aug_path + '\\' + aug_list[i]
    test_name = test_path + '\\' + test_list[i]
    aug_df = pd.read_csv(aug_name)
    test_df = pd.read_csv(test_name)

    empty_df = empty_df.append(test_df)

print(empty_df)

# for i in range(len(aug_list)+1):
#     print("present :", start)
#
#     aug_name = aug_path + '\\' + aug_list[i]
#     test_name = test_path + '\\' + test_list[i]
#     aug_df = pd.read_csv(aug_name)
#     test_df = pd.read_csv(test_name)