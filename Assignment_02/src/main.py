import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import ExtraTreesClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz # install graphviz using 'pip3 install graphviz' and binaries for graphviz using 'brew install graphviz'
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

# train_df.info()
# test_df.info()

# Check for all missing values in the training set 
# print(train_df.isnull().sum())

# *** PREPROCESSING TRAINING DATASET ***

# Fill null values in the Age feature using KNN algorithm, K = 5
X = train_df['Age'].values
X = X.reshape(-1,1)

imputer = KNNImputer(n_neighbors=5)
Xtrans = imputer.fit_transform(X)

# print before and after
for i, x in enumerate(X):
	X[i][0], Xtrans[i][0]
	# print(X[i][0], Xtrans[i][0])

pd.DataFrame(Xtrans).shape

# Store Xtrans in the first column of train_df
train_df = pd.concat([pd.DataFrame(Xtrans), train_df], axis = 1)

# Delete the 'Age' column of train_df.
del train_df['Age']

# Rename the the 'Xtrans' column as 'Age'
train_df.rename(columns = {0: 'Age'}, inplace = True)

# print("After KNN algorithm, The number of null values in the Age feature are: ", np.sum(train_df.Age == np.nan))

# Fill null values in the Embarked feature using mode, most common occurance.

Embarked_mode = train_df['Embarked'].mode()
train_df['Embarked'].fillna(Embarked_mode[0], inplace = True)

# print("In the training dataset, The mode for Embarked freature is: ", Embarked_mode)
# print("The number of empty values in Embarked feature is: ", train_df['Embarked'].isnull().sum())
# print("After filling the empty values, the number of empty values in the Embarked feature is: ", train_df['Embarked'].isnull().sum())
# print("The following are the values in the Embarked feature: ", train_df['Embarked'])

# Map the Sex feature in the dataset to binary values: female = 1 and male = 0

train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male':0})
# print("Sex values: \n", train_df['Sex'])

# Map the Embarked feature in the datset to binary values: S = 0, C = 1 and Q = 2

train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C':1, 'Q':2})
# print("Embarked values: \n", train_df['Embarked'])

# Binning Fare feature

train_df['Fare_bin'] = pd.cut(train_df['Fare'], [-0.001, 7.91, 14.454, 31.0, 512.3292], labels=['0', '1', '2', '3'])
# print("Fare Bin values: \n", train_df['Fare_bin'])
# print(train_df.head())

# *** FEATURE ENGINEERING ***

# Lets drop the Cabin feature as it has a lot of missing values. 

train_df = train_df.drop(['Cabin'], axis = 1)

# Lets drop the Ticket feature as it does not seem to provide any vital information for our analysis.

train_df = train_df.drop(['Ticket'], axis = 1)

# Lets drop the Name feature as it does not seem to provide any vital information for our analysis.

train_df = train_df.drop(['Name'], axis = 1)

# Lets drop the Fare feature as we have performed binning based on our Fare values

train_df = train_df.drop(['Fare'], axis = 1)

# Lets drop the PassengerId feature as it does not provide any vital information for our analysis

train_df = train_df.drop(['PassengerId'], axis = 1)

# Remaining features are:

# train_df.info()

# Check for any null columns before starting the feature selection

# print(train_df.isnull().sum())
# null_columns=train_df.columns[train_df.isnull().any()]
# print(train_df[null_columns].isnull().sum())
# print(train_df[train_df["Fare_bin"].isnull()][null_columns])

# *** FEATURE SELECTION USING SELECTKBEST

X = train_df.drop("Survived", axis = 1)
y = train_df["Survived"]

mdlsel = SelectKBest(score_func=chi2, k=5)
mdlsel.fit(X, y)
ix = mdlsel.get_support()
data = pd.DataFrame(mdlsel.transform(X), index=None, columns=X.columns.values[ix], dtype=None, copy=False)
# print(data.head(n=5))

# *** FEATURE SELECTION USING EXTRA TREE CLASSIFIERS

model = ExtraTreesClassifier(criterion="gini")
model.fit(X, y, sample_weight=None)
# print(model.feature_importances_)
# print(X.head())

# *** DECISION TREE MODEL ***

# Create a feature list

target = train_df['Survived'].values
data_features = ['Age', 'Pclass', 'Sex', 'Fare_bin', 'Embarked', 'SibSp', 'Parch']
features = train_df[data_features].values

# Build training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 52)

# Build the tree

decision_tree = tree.DecisionTreeClassifier(criterion="gini", max_depth=None, min_samples_split=100, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=52, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None)
decision_tree_training = decision_tree.fit(X_train, y_train)
target_predict = decision_tree_training.predict(X_test)

# Check the accuracy score of the decision tree

print("\nDecision tree accuracy score is: ", accuracy_score(y_test, target_predict), end = "\n\n")

# *** PLOT DECISION TREE MODEL ***

generalized_tree = tree.DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=5, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0., max_features=None, random_state=52, max_leaf_nodes=None, min_impurity_decrease=0., min_impurity_split=None, class_weight=None)
generalized_tree_fit = generalized_tree.fit(features, target)

# Check the accuracy score of the generalized decision tree

print("Generalized tree accuracy score is: ", generalized_tree_fit.score(features, target), end = "\n\n")

graph_data = tree.export_graphviz(generalized_tree, feature_names= data_features, out_file = None)
graph = graphviz.Source(graph_data)
graph.view()


# *** RANDOM FOREST MODEL ***

target = train_df['Survived'].values
data_features = ['Age', 'Pclass', 'Sex', 'Fare_bin', 'Embarked', 'SibSp', 'Parch']
features = train_df[data_features].values

# Build training and testing dataset

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 52)

# Build the tree

random_forest = RandomForestClassifier(n_estimators=100, criterion="gini", max_depth=5, min_samples_split=2, min_samples_leaf=1, max_leaf_nodes=None, random_state=40)
random_forest_fit = random_forest.fit(X_train, y_train, sample_weight=None)
target_predict = random_forest_fit.predict(X_test)

# Check the accuracy score of the Random Forest tree

print("The accuracy score of the random forest tree is: ", accuracy_score(y_test, target_predict), end = "\n\n")

# *** Five-fold cross validation on Decision Tree algorithm

decision_tree_CV = DecisionTreeClassifier(min_samples_split = 20, random_state = 20)
decision_tree_CV.fit(X_train, y_train)
scores_DT = cross_val_score(decision_tree_CV, X_train, y_train, cv = 5)
print("Five-fold cross validation on Decision Tree algorithm, Mean and standard deviation is: " , scores_DT.mean(), scores_DT.std(), end = "\n\n")

# *** Five-fold cross validation on Random Forest algorithm

random_forest_CV = RandomForestClassifier(min_samples_split = 20, random_state = 20)
random_forest_CV.fit(X_train, y_train)
scores_RF = cross_val_score(random_forest_CV, X_train, y_train, cv = 5)
print("Five-fold cross validation on Random Forest algorithm, Mean and standard deviation is: " , scores_RF.mean(), scores_RF.std(), end = "\n\n")













