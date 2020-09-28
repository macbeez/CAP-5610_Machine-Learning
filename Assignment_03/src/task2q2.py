import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from progress.bar import Bar
from tabulate import tabulate
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split

train_titanic = pd.read_csv("include/train1.csv")
test_titanic = pd.read_csv("include/test1.csv")
results_titanic = pd.read_csv("include/gender_submission.csv")

# train_titanic.info()
# test_titanic.info()

##################################################################
# *** KNN FROM SCRATCH ***
##################################################################

# Feature engineering

# Drop Cabin features as it seems to have too many unknown values
train_titanic = train_titanic.drop(['Cabin'], axis = 1)
test_titanic = test_titanic.drop(['Cabin'], axis = 1)
train_titanic = train_titanic.drop(['Name'], axis = 1)
test_titanic = test_titanic.drop(['Name'], axis = 1)
train_titanic = train_titanic.drop(['Ticket'], axis = 1)
test_titanic = test_titanic.drop(['Ticket'], axis = 1)

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

accuracy = []
from knn import KNN

#### K Fold Validation ####

max_knn = 25
max_split = 5

bar = Bar('Processing', max = max_knn*max_split)

knn_val = 1
while knn_val <= max_knn:

    cv = KFold(n_splits=max_split, random_state=None, shuffle=False)
    split_index = 1
    kfold_acc = []
    for train_index, test_index in cv.split(X_train):
        # Break up X_train and y_train in to test/train datasets for CV
        X_cv_train, y_cv_train = X_train[train_index], y_train[train_index]
        X_cv_test, y_cv_test = X_train[test_index], y_train[test_index]
        knn_clf = KNN(k=knn_val)
        knn_clf.fit(X_cv_train, y_cv_train)
        y_cv_pred = knn_clf.predict(X_cv_test)

        acc = np.sum(y_cv_pred == y_cv_test)/len(y_cv_test)
        kfold_acc.append(acc)
        # print("KNN:", knn_val, "KFold Split:", split_index, "Accuracy:", acc)
        bar.next()
        split_index += 1

    accuracy.append([knn_val, sum(kfold_acc)/len(kfold_acc)])

    knn_val = knn_val + 1
bar.finish()

print(tabulate(accuracy, headers=["k", "Accuracy"], tablefmt='psql'))

K_values = [row[0] for row in accuracy]
Acc_values = [row[1] for row in accuracy]

max_accuracy = max(Acc_values)
max_accuracy_knn = K_values[Acc_values.index(max_accuracy)]

print("\nKNN Value with best accuracy:")
print("K=", max_accuracy_knn, "Accuracy=", "{:.2f}%".format(max_accuracy*100), end="\n\n")

# Plot the graph
plt.plot(K_values, Acc_values)
plt.xlabel("K Values")
plt.ylabel("Accuracy Values")
plt.title("Accuracy vs. K values")
plt.show()

## KNN on the entire dataset ###
knn_clf = KNN(k = max_accuracy_knn)
knn_clf.fit(X_train, y_train)
y_pred = knn_clf.predict(X_test)

print("KNN METRICS:")
print("\nThe accuracy on the entire model is: %.4f" % metrics.accuracy_score(y_test, y_pred))
print("The precision on the entire model is: %.4f" % metrics.precision_score(y_test, y_pred))
print("The recall on the entire model is: %.4f" % metrics.recall_score(y_test, y_pred))
print("The f1 score on the entire model is: %.4f" % metrics.f1_score(y_test, y_pred))










