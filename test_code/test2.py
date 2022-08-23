import pandas as pd

amazon_data = '/Users/ruin/Desktop/data/amazon_review_polarity_csv/test.csv'

amazon_data = pd.read_csv(amazon_data)

print(amazon_data.columns)