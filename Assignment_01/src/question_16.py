import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male':0})
print(train_df['Sex'])