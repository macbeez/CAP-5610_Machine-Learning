import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")


print(train_df.isnull().sum())
print(test_df.isnull().sum())

combine_cabin = train_df['Cabin'].append(test_df['Cabin'])
# print(combine_cabin)

print("Out of ", len(combine_cabin), "entries in the cabin feature, the number of null values in Cabin feature are: ", combine_cabin.isnull().sum(), "entries.")
