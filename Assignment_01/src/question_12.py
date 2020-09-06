import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

train_df_survived = train_df[train_df["Survived"] == 1]
train_df_died = train_df[train_df["Survived"] == 0]

train_df_Pclass1 = train_df[train_df["Pclass"] == 1]
train_df_Pclass2 = train_df[train_df["Pclass"] == 2]
train_df_Pclass3 = train_df[train_df["Pclass"] == 3]

train_df_Pc1_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Pclass"] == 1)]
train_df_Pc2_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Pclass"] == 2)]
train_df_Pc3_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Pclass"] == 3)]

train_df_Pc1_Died = train_df[(train_df["Survived"] == 0) & (train_df["Pclass"] == 1)]
train_df_Pc2_Died = train_df[(train_df["Survived"] == 0) & (train_df["Pclass"] == 2)]
train_df_Pc3_Died = train_df[(train_df["Survived"] == 0) & (train_df["Pclass"] == 3)]

f, axes = plt.subplots(3, 2, figsize=(7, 7), sharey = True)

sns.distplot(train_df_Pc1_Died['Age'], color = 'g', kde = False, ax = axes[0,0])
axes[0,0].set_title("Pclass = 1 | Survived = 0")
axes[0,0].set_xlim(0,80)

sns.distplot(train_df_Pc1_Sur['Age'], color = 'g', kde = False, ax = axes[0,1])
axes[0,1].set_title("Pclass = 1 | Survived = 1")
axes[0,1].set_xlim(0,80)

sns.distplot(train_df_Pc2_Died['Age'], color = 'r', kde = False, ax = axes[1,0])
axes[1,0].set_title("Pclass = 2 | Survived = 0")
# axes[1,0].title.set_position([.5, 1.1])
axes[1,0].set_xlim(0,80)

sns.distplot(train_df_Pc2_Sur['Age'], color = 'r', kde = False, ax = axes[1,1])
axes[1,1].set_title("Pclass = 2 | Survived = 1")
axes[1,1].set_xlim(0,80)

sns.distplot(train_df_Pc3_Died['Age'], color = 'b', kde = False, ax = axes[2,0])
axes[2,0].set_title("Pclass = 3 | Survived = 0")
axes[2,0].set_xlim(0,80)

sns.distplot(train_df_Pc3_Sur['Age'], color = 'b', kde = False, ax = axes[2,1])
axes[2,1].set_title("Pclass = 3 | Survived = 1")
axes[2,1].set_xlim(0,80)

plt.tight_layout()
plt.show()