import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

train_df["Survived"] = train_df["Survived"].astype("category")
train_df['Pclass'] = train_df['Pclass'].astype('category')
train_df['Sex'] = train_df['Sex'].astype('category')
train_df['Embarked'] = train_df['Embarked'].astype('category')

print(train_df.describe(include="category"))