import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

print("In the training dataset, The mode for Embarked freature is: ", train_df['Embarked'].mode())
print("The number of empty values in Embarked feature is: ", train_df['Embarked'].isnull().sum())
train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
print("After filling the empty values, the number of empty values in the Embarked feature is: ", train_df['Embarked'].isnull().sum())
print("The following are the values in the Embarked feature: ", train_df['Embarked'])
