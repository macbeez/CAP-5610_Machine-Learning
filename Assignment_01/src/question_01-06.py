import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

train_df.info()
test_df.info()

print(train_df.isnull().sum())
print(test_df.isnull().sum())
