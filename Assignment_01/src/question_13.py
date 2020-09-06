
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

# train_df['Sex'] = train_df['Sex'].astype('category')
# print(train_df.describe(include='category'))

train_df_S_Died = train_df[(train_df["Survived"] == 0) & (train_df["Embarked"] == 'S')]
train_df_S_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Embarked"] == 'S')]
train_df_C_Died = train_df[(train_df["Survived"] == 0) & (train_df["Embarked"] == 'C')]
train_df_C_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Embarked"] == 'C')]
train_df_Q_Died = train_df[(train_df["Survived"] == 0) & (train_df["Embarked"] == 'Q')]
train_df_Q_Sur = train_df[(train_df["Survived"] == 1) & (train_df["Embarked"] == 'Q')]

f, axes = plt.subplots(3, 2, figsize=(7, 7), sharex = True, sharey = True)

sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_S_Died, ax = axes[0,0], ci=None)
axes[0,0].set_title("Embarked = S | Survived = 0")
axes[0,0].set_ylabel("Fare")
axes[0,0].set_ylim(0,80)
axes[0,0].get_legend().set_visible(False)


sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_S_Sur, ax = axes[0,1], ci=None)
axes[0,1].set_title("Embarked = S | Survived = 1")
# axes[0,1].set_ylabel("Fare")
axes[0,1].set_ylim(0,80)
axes[0,1].get_legend().set_visible(False)

sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_C_Died, ax = axes[1,0], ci=None)
axes[1,0].set_title("Embarked = C | Survived = 0")
axes[1,0].set_ylabel("Fare")
axes[1,0].set_ylim(0,80)
axes[1,0].get_legend().set_visible(False)


sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_C_Sur, ax = axes[1,1], ci=None)
axes[1,1].set_title("Embarked = C | Survived = 1")
# axes[1,1].set_ylabel("Fare")
axes[1,1].set_ylim(0,80)
axes[1,1].get_legend().set_visible(False)

sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_Q_Died, ax = axes[2,0], ci=None)
axes[2,0].set_title("Embarked = Q | Survived = 0")
axes[2,0].set_ylabel("Fare")
axes[2,0].set_ylim(0,80)
axes[2,0].get_legend().set_visible(False)

sns.barplot(x = 'Sex', y = 'Fare', hue = 'Survived', data = train_df_Q_Sur, ax = axes[2,1], ci=None)
axes[2,1].set_title("Embarked = Q | Survived = 1")
# axes[2,1].set_ylabel("Fare")
axes[2,1].set_ylim(0,80)
axes[2,1].get_legend().set_visible(False)

plt.tight_layout()
plt.show()