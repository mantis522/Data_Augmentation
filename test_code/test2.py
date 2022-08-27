import pandas as pd
import numpy as np

test_dir = r"D:\ruin\data\amazon\amazon_review_polarity_csv\test2.csv"
test_data = pd.read_csv(test_dir)

df_shuffled = test_data.iloc[np.random.permutation(test_data.index)].reset_index(drop=True)

df_shuffled.to_csv("test2.csv", index=False)