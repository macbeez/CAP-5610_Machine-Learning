
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
from sklearn import metrics

train_titanic = pd.read_csv("include/train1.csv")
test_titanic = pd.read_csv("include/test1.csv")
results_titanic = pd.read_csv("include/gender_submission.csv")


###############################################################
### TASK 2
###############################################################

# Feature engineering

# Drop Cabin features as it seems to have too many unknown values
train_titanic = train_titanic.drop(['Cabin'], axis = 1)
test_titanic = test_titanic.drop(['Cabin'], axis = 1)
train_titanic = train_titanic.drop(['Name'], axis = 1)
test_titanic = test_titanic.drop(['Name'], axis = 1)
train_titanic = train_titanic.drop(['Ticket'], axis = 1)
test_titanic = test_titanic.drop(['Ticket'], axis = 1)
#results_titanic = results_titanic.drop(['PassengerId'], axis = 1)

# Fill null values in the Embarked feature using mode, most common occurance.

Embarked_mode = train_titanic['Embarked'].mode()
train_titanic['Embarked'].fillna(Embarked_mode[0], inplace = True)

# Fill null values in the Age feature using mode, most common occurance.

Age_mode_train = train_titanic['Age'].mode()
train_titanic['Age'].fillna(Age_mode_train[0], inplace = True)

Age_mode_test = test_titanic['Age'].mode()
test_titanic['Age'].fillna(Age_mode_test[0], inplace = True)

# Fill the only null value of the Fare feature inn testing set with the mode
Fare_mode_test = test_titanic['Fare'].mode()
test_titanic['Fare'].fillna(Fare_mode_test[0], inplace = True)

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.

# train_titanic['Name'] = le.fit_transform(train_titanic['Name'])
train_titanic['Sex'] = le.fit_transform(train_titanic['Sex'])
train_titanic['Age'] = le.fit_transform(train_titanic['Age'])
# train_titanic['Ticket'] = le.fit_transform(train_titanic['Ticket'])
train_titanic['Fare'] = le.fit_transform(train_titanic['Fare'])
train_titanic['Embarked'] = le.fit_transform(train_titanic['Embarked'])

# test_titanic['Name'] = le.fit_transform(test_titanic['Name'])
test_titanic['Sex'] = le.fit_transform(test_titanic['Sex'])
test_titanic['Age'] = le.fit_transform(test_titanic['Age'])
# test_titanic['Ticket'] = le.fit_transform(test_titanic['Ticket'])
test_titanic['Fare'] = le.fit_transform(test_titanic['Fare'])
test_titanic['Embarked'] = le.fit_transform(test_titanic['Embarked'])

data_features = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = train_titanic['Survived'].values
features = train_titanic[data_features].values

X_train = train_titanic[data_features].values
y_train = train_titanic['Survived'].values
X_test = test_titanic[data_features].values
y_test = results_titanic['Survived'].values

# results_titanic = results_titanic.drop(['PassengerId'], axis = 1)

# *** NAIVE BAYES MODEL ***

### K Fold Cross Validation with K = 5
cv = KFold(n_splits=5, random_state=None, shuffle=False)

# Create a Gaussian Naive Bayes Classifier
clf = GaussianNB()

i = 1
accuracy, precision, recall, f1_score = [], [], [], []

for train_index, test_index in cv.split(X_train):
    # print("\niteration", i, ":")

    # Break up X_train and y_train in to test/train datasets for CV
    X_cv_train, y_cv_train = X_train[train_index], y_train[train_index]
    X_cv_test, y_cv_test = X_train[test_index], y_train[test_index]

    # print("Size of training dataset", len(train_index))
    # print("Size of testing dataset ", len(test_index))

    clf.fit(X_cv_train, y_cv_train)
    y_cv_pred = clf.predict(X_cv_test)
    # print(len(y_test))
    # print(len(y_cv_pred))

    # Print just the first 10 to check the results, actual score should be based on the whole array
    # l = 5
    # print("Actual result for given index     ", test_index[:l], "are:", y_cv_test[:l])
    # print("Predicted result for a given index", test_index[:l], "are:", y_cv_pred[:l])
    
    accuracy.append(metrics.accuracy_score(y_cv_test, y_cv_pred))
    precision.append(metrics.precision_score(y_cv_test, y_cv_pred))
    recall.append(metrics.recall_score(y_cv_test, y_cv_pred))
    f1_score.append(metrics.f1_score(y_cv_test, y_cv_pred))

    i = i+1
    # print("=====================================")
print("Average metrics over five folds: ")
print("\nThe average accuracy is: %.4f" % (sum(accuracy)/len(accuracy)))
print("The average precision is: %.4f" % (sum(precision)/len(precision)))
print("The average recall is: %.4f" % (sum(recall)/len(recall)))
print("The average f1_score is: %.4f" % (sum(f1_score)/len(f1_score)))

# fit and predict on the entire model
clf.fit(X_train, y_train)
y_predict = clf.predict(X_test)
print("\nNAIVE BAYES METRICS:")
print("\nThe accuracy on the entire model is: %.4f" % metrics.accuracy_score(y_test, y_predict))
print("The precision on the entire model is: %.4f" % metrics.precision_score(y_test, y_predict))
print("The recall on the entire model is: %.4f" % metrics.recall_score(y_test, y_predict))
print("The f1 score on the entire model is: %.4f" % metrics.f1_score(y_test, y_predict))










