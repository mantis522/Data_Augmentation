import os
import pandas as pd

test_list = ['date', 'numbers', 'acc', 'loss', 'recall',
             'precision', 'F1_micro', 'F1_macro']

if os.path.isfile('test2.csv'):
    print("Yes. it is a file")

else:
    test_list = ['date', 'numbers', 'acc', 'loss', 'recall',
                 'precision', 'F1_micro', 'F1_macro']
    df_making = pd.DataFrame(columns=test_list)
    df_making.to_csv('test2.csv', index=False)