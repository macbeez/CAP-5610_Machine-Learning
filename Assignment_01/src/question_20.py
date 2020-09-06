import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

train_df['Fare_bin'] = pd.cut(train_df['Fare'], [-0.001, 7.91, 14.454, 31.0, 512.329], labels=['0', '1', '2', '3'])
print(train_df.head())