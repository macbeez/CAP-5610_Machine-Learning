import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

plt.figure(figsize = (10,8))
cht = sns.heatmap(train_df.corr(method='pearson'))
cht.set_xticklabels(cht.get_xmajorticklabels(), fontsize = 9)
cht.set_yticklabels(cht.get_ymajorticklabels(), fontsize = 9)
cht.invert_yaxis()
plt.show()

print(pd.crosstab(train_df['Pclass'], train_df['Survived']))
plt.figure()
sns.countplot(x = 'Pclass', hue = 'Survived', data = train_df)
plt.show()

print(pd.crosstab(train_df['Sex'], train_df['Survived']))
plt.figure()
sns.countplot(x = 'Sex', hue = 'Survived', data = train_df)
plt.show()