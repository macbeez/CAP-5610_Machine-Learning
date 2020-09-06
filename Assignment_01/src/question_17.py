import pandas as pd
from sklearn.impute import KNNImputer
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

X = train_df['Age'].values
X = X.reshape(-1,1)

imputer = KNNImputer(n_neighbors=5)
Xtrans = imputer.fit_transform(X)

# print before and after
for i, x in enumerate(X):
	X[i][0], Xtrans[i][0]
	print(X[i][0], Xtrans[i][0])

pd.DataFrame(Xtrans).shape

# Store Xtrans in the first column of train_df
train_df = pd.concat([pd.DataFrame(Xtrans), train_df], axis = 1)

# Delete the 'Age' column of train_df.
del train_df['Age']

# Rename the the 'Xtrans' column as 'Age'
train_df.rename(columns = {0: 'Age'}, inplace = True)

print("The number of Nan values in the Age column is: ", np.sum(train_df.Age == np.nan))