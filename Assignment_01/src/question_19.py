import pandas as pd
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

print("In the testing dataset, The mode for Fare freature is: ", test_df['Fare'].mode())
print("The number of empty values in Fare feature is: ", test_df['Fare'].isnull().sum())
test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)
print("After filling the empty value, the number of empty values in the Fare feature is: ", test_df['Fare'].isnull().sum())
print("The following are the values in the Fare feature: ", test_df['Fare'])
