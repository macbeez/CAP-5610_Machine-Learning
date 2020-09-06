import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

train_df_survived = train_df[train_df["Survived"] == 1]
train_df_died = train_df[train_df["Survived"] == 0]

f, axes = plt.subplots(1, 2, figsize=(7, 7), sharey=True)
sns.distplot(train_df_survived['Age'], color='g', kde = False, ax=axes[0])
axes[0].set_title("Survived = 1")
axes[0].set_xlim(0,80)

sns.distplot(train_df_died['Age'], color='r', kde = False, ax=axes[1])
axes[1].set_title("Survived = 0")
axes[1].set_xlim(0,80)
plt.show()

age_80 = train_df[train_df["Age"] == 80]
print(age_80)
print()
print()

mid_age_s = train_df[(train_df["Age"] >=15) & (train_df["Age"] <= 25) & (train_df["Survived"] == 1)]
print(mid_age_s.describe())
print()
print()

mid_age_d = train_df[(train_df["Age"] >=15) & (train_df["Age"] <= 25) & (train_df["Survived"] == 0)]
print(mid_age_d.describe())