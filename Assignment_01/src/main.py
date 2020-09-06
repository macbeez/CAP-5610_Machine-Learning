import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")
# combine_df = [train_df, test_df]

# *** QUESTION 1 - 6 ***

# train_df.info()
# test_df.info()

# print(train_df.isnull().sum())

# *** QUESTION 7 ***
# print(train_df.describe())

# *** QUESTION 8 ***

# train_df["Survived"] = train_df["Survived"].astype("category")
# train_df['Pclass'] = train_df['Pclass'].astype('category')
# train_df['Sex'] = train_df['Sex'].astype('category')
# train_df['Embarked'] = train_df['Embarked'].astype('category')

# print(train_df.describe(include="category"))


# *** QUESTION 9 AND 10 ***
# plt.figure(figsize = (10,8))
# cht = sns.heatmap(train_df.corr(method='pearson'))
# cht.set_xticklabels(cht.get_xmajorticklabels(), fontsize = 9)
# cht.set_yticklabels(cht.get_ymajorticklabels(), fontsize = 9)
# cht.invert_yaxis()
# plt.show()

# print(pd.crosstab(train_df['Pclass'], train_df['Survived']))
# plt.figure()
# sns.countplot(x = 'Pclass', hue = 'Survived', data = train_df)
# plt.show()

# print(pd.crosstab(train_df['Sex'], train_df['Survived']))
# plt.figure()
# sns.countplot(x = 'Sex', hue = 'Survived', data = train_df)
# plt.show()


# *** Question 11 ***

# train_df_survived = train_df[train_df["Survived"] == 1]
# train_df_died = train_df[train_df["Survived"] == 0]

# f, axes = plt.subplots(1, 2, figsize=(7, 7), sharey=True)
# sns.distplot(train_df_survived['Age'], color='g', kde = False, ax=axes[0])
# axes[0].set_title("Survived = 1")
# axes[0].set_xlim(0,80)

# sns.distplot(train_df_died['Age'], color='r', kde = False, ax=axes[1])
# axes[1].set_title("Survived = 0")
# axes[1].set_xlim(0,80)
# plt.show()

# age_80 = train_df[train_df["Age"] == 80]
# print(age_80)
# print()
# print()

# mid_age_s = train_df[(train_df["Age"] >=15) & (train_df["Age"] <= 25) & (train_df["Survived"] == 1)]
# print(mid_age_s.describe())
# print()
# print()

# mid_age_d = train_df[(train_df["Age"] >=15) & (train_df["Age"] <= 25) & (train_df["Survived"] == 0)]
# print(mid_age_d.describe())


# *** Question 12 ***
# train_df_survived = train_df[train_df["Survived"] == 1]
# train_df_died = train_df[train_df["Survived"] == 0]

# train_df_Pclass1 = train_df[train_df["Pclass"] == 1]
# train_df_Pclass2 = train_df[train_df["Pclass"] == 2]
# train_df_Pclass3 = train_df[train_df["Pclass"] == 3]

# train_df_Pc1_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Pclass"] == 1)]
# train_df_Pc2_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Pclass"] == 2)]
# train_df_Pc3_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Pclass"] == 3)]

# train_df_Pc1_Died = train_df[(train_df["Survived"] == 0) & (train_df["Pclass"] == 1)]
# train_df_Pc2_Died = train_df[(train_df["Survived"] == 0) & (train_df["Pclass"] == 2)]
# train_df_Pc3_Died = train_df[(train_df["Survived"] == 0) & (train_df["Pclass"] == 3)]

# f, axes = plt.subplots(3, 2, figsize=(7, 7), sharey = True)

# sns.distplot(train_df_Pc1_Died['Age'], color = 'g', kde = False, ax = axes[0,0])
# axes[0,0].set_title("Pclass = 1 | Survived = 0")
# axes[0,0].set_xlim(0,80)

# sns.distplot(train_df_Pc1_Sur['Age'], color = 'g', kde = False, ax = axes[0,1])
# axes[0,1].set_title("Pclass = 1 | Survived = 1")
# axes[0,1].set_xlim(0,80)

# sns.distplot(train_df_Pc2_Died['Age'], color = 'r', kde = False, ax = axes[1,0])
# axes[1,0].set_title("Pclass = 2 | Survived = 0")
# # axes[1,0].title.set_position([.5, 1.1])
# axes[1,0].set_xlim(0,80)

# sns.distplot(train_df_Pc2_Sur['Age'], color = 'r', kde = False, ax = axes[1,1])
# axes[1,1].set_title("Pclass = 2 | Survived = 1")
# axes[1,1].set_xlim(0,80)

# sns.distplot(train_df_Pc3_Died['Age'], color = 'b', kde = False, ax = axes[2,0])
# axes[2,0].set_title("Pclass = 3 | Survived = 0")
# axes[2,0].set_xlim(0,80)

# sns.distplot(train_df_Pc3_Sur['Age'], color = 'b', kde = False, ax = axes[2,1])
# axes[2,1].set_title("Pclass = 3 | Survived = 1")
# axes[2,1].set_xlim(0,80)

# plt.tight_layout()
# plt.show()

# *** QUESTION 13 ***
# train_df['Sex'] = train_df['Sex'].astype('category')

# train_df_S_Died = train_df[(train_df["Survived"] == 0) & (train_df["Embarked"] == 'S')]
# train_df_S_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Embarked"] == 'S')]
# train_df_C_Died = train_df[(train_df["Survived"] == 0) & (train_df["Embarked"] == 'C')]
# train_df_C_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Embarked"] == 'C')]
# train_df_Q_Died = train_df[(train_df["Survived"] == 0) & (train_df["Embarked"] == 'Q')]
# train_df_Q_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Embarked"] == 'Q')]

# f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex = True, sharey = True)

# sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_S_Died, ax = axes[0,0], ci=None)
# axes[0,0].set_title("Embarked = S | Survived = 0")
# axes[0,0].set_ylabel("Fare")
# axes[0,0].set_ylim(0,80)
# axes[0,0].get_legend().set_visible(False)


# sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_S_Sur, ax = axes[0,1], ci=None)
# axes[0,1].set_title("Embarked = S | Survived = 1")
# # axes[0,1].set_ylabel("Fare")
# axes[0,1].set_ylim(0,80)
# axes[0,1].get_legend().set_visible(False)

# sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_C_Died, ax = axes[1,0], ci=None)
# axes[1,0].set_title("Embarked = C | Survived = 0")
# axes[1,0].set_ylabel("Fare")
# axes[1,0].set_ylim(0,80)
# axes[1,0].get_legend().set_visible(False)


# sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_C_Sur, ax = axes[1,1], ci=None)
# axes[1,1].set_title("Embarked = C | Survived = 1")
# # axes[1,1].set_ylabel("Fare")
# axes[1,1].set_ylim(0,80)
# axes[1,1].get_legend().set_visible(False)

# sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_Q_Died, ax = axes[2,0], ci=None)
# axes[2,0].set_title("Embarked = Q | Survived = 0")
# axes[2,0].set_ylabel("Fare")
# axes[2,0].set_ylim(0,80)
# axes[2,0].get_legend().set_visible(False)

# sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_Q_Sur, ax = axes[2,1], ci=None)
# axes[2,1].set_title("Embarked = Q | Survived = 1")
# # axes[2,1].set_ylabel("Fare")
# axes[2,1].set_ylim(0,80)
# axes[2,1].get_legend().set_visible(False)

# plt.tight_layout()
# plt.show()

# *** QUESTION 14 ***

# unique_ticket = train_df['Ticket'].unique()
# # print(unique_ticket)
# print("Total number of ticket entries ", len(train_df['Ticket']))
# print("Number of unique ticket values ", len(unique_ticket))

# # rate of duplicates = (total records - unique records)/total records

# Ticket_dup_rate = (len(train_df['Ticket']) - len(unique_ticket)) * 100 /len(train_df['Ticket'])
# print("Total ticket duplicate rate is ", Ticket_dup_rate, "%")

# *** QUESTION 15 ***

# print(train_df.isnull().sum())
# print(test_df.isnull().sum())

# combine_cabin = train_df['Cabin'].append(test_df['Cabin'])
# # print(combine_cabin)

# print("Out of ", len(combine_cabin), "entries in the cabin feature, the number of null values in Cabin feature are: ", combine_cabin.isnull().sum(), "entries.")

# # *** QUESTION 16 ***
# train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male':0})
# print(train_df['Sex'])

# *** QUESTION 17 ***

# X = train_df['Age'].values
# X = X.reshape(-1,1)

# imputer = KNNImputer(n_neighbors=5)
# Xtrans = imputer.fit_transform(X)

# # print before and after
# for i, x in enumerate(X):
# 	print(X[i][0], Xtrans[i][0])

# pd.DataFrame(Xtrans).shape

# # Store Xtrans in the first column of train_df
# train_df = pd.concat([pd.DataFrame(Xtrans), train_df], axis = 1)

# # Delete the 'Age' column of train_df.
# del train_df['Age']

# # Rename the the 'Xtrans' column as 'Age'
# train_df.rename(columns = {0: 'Age'}, inplace = True)

# print(np.sum(train_df.Age == np.nan))


# *** QUESTION 18 ***

# print("In the training dataset, The mode for Embarked freature is: ", train_df['Embarked'].mode())
# print("The number of empty values in Embarked feature is: ", train_df['Embarked'].isnull().sum())
# train_df['Embarked'].fillna(train_df['Embarked'].mode()[0], inplace = True)
# print("After filling the empty values, the number of empty values in the Embarked feature is: ", train_df['Embarked'].isnull().sum())
# print("The following are the values in the Embarked feature: ", train_df['Embarked'])

# *** QUESTION 19 ***

# print("In the testing dataset, The mode for Fare freature is: ", test_df['Fare'].mode())
# print("The number of empty values in Fare feature is: ", test_df['Fare'].isnull().sum())
# test_df['Fare'].fillna(test_df['Fare'].mode()[0], inplace = True)
# print("After filling the empty value, the number of empty values in the Fare feature is: ", test_df['Fare'].isnull().sum())
# print("The following are the values in the Fare feature: ", test_df['Fare'])


# *** QUESTION 20 ***

train_df['Fare_bin'] = pd.cut(train_df['Fare'], [-0.001, 7.91, 14.454, 31.0, 512.329], labels=['0', '1', '2', '3'])
print(train_df.head())








