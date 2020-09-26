import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import nltk # needed for Naive-Bayes
import numpy as np
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
from tabulate import tabulate
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from progress.bar import Bar


train_df = pd.read_csv("include/train2.csv")
test_df = pd.read_csv("include/test2.csv")

train_titanic = pd.read_csv("include/train1.csv")
test_titanic = pd.read_csv("include/test1.csv")
results_titanic = pd.read_csv("include/gender_submission.csv")

# train_df.info()
# test_df.info()

# train_titanic.info()
# test_titanic.info()

###############################################################
### TASK 1
###############################################################

# Creating labelEncoder
le = preprocessing.LabelEncoder()

# Converting string labels into numbers.
train_df['Date'] = le.fit_transform(train_df['Date'])
train_df['Opponent'] = le.fit_transform(train_df['Opponent'])
train_df['Is_Home_or_Away'] = le.fit_transform(train_df['Is_Home_or_Away'])
train_df['Is_Opponent_in_AP25_Preseason'] = le.fit_transform(train_df['Is_Opponent_in_AP25_Preseason'])
train_df['Media'] = le.fit_transform(train_df['Media'])

test_df['Date'] = le.fit_transform(test_df['Date'])
test_df['Opponent'] = le.fit_transform(test_df['Opponent'])
test_df['Is_Home_or_Away'] = le.fit_transform(test_df['Is_Home_or_Away'])
test_df['Is_Opponent_in_AP25_Preseason'] = le.fit_transform(test_df['Is_Opponent_in_AP25_Preseason'])
test_df['Media'] = le.fit_transform(test_df['Media'])

train_df['Label'] = le.fit_transform(train_df['Label'])
test_df['Label'] = le.fit_transform(test_df['Label'])

# target = train_df['Label'].values
data_features = ['Date', 'Opponent', 'Is_Home_or_Away', 'Is_Opponent_in_AP25_Preseason', 'Media']
# features = train_df[data_features].values

X_train = train_df[data_features].values
y_train = train_df['Label'].values
X_test = test_df[data_features].values
y_test = test_df['Label'].values
# print("Given y labels: ", y_test)

# print(X_train, y_train, X_test, y_pred)

# *** NAIVE BAYES MODEL ***

# Create a Gaussian Naive Bayes Classifier
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(X_train, y_train)

# Predict the response for test_titanic dataset
y_pred = gnb.predict(X_test)
# print("Naive Bayes: Predicted y labels: ", y_pred)

# Model Accuracy, how often is the classifier correct?
NB_accuracy = metrics.accuracy_score(y_test, y_pred)
NB_precision = metrics.precision_score(y_test, y_pred)
NB_recall = metrics.recall_score(y_test, y_pred)
NB_f1_score = metrics.f1_score(y_test, y_pred)

# *** KNN MODEL ***

# Create a KNN Classifier
knn = KNeighborsClassifier(n_neighbors = 3)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for the test_titanic dataset
y_pred = knn.predict(X_test)
# print("KNN: Predicted y labels: ", y_pred)

# Model Accuracy, how often is the classifier correct?
KNN_accuracy = metrics.accuracy_score(y_test, y_pred)
KNN_precision = metrics.precision_score(y_test, y_pred)
KNN_recall = metrics.recall_score(y_test, y_pred)
KNN_f1_score = metrics.f1_score(y_test, y_pred)

# Compare Accuracy, Precision, Recall and F1 Score between Naive Bayes and KNN

print("\nNaive Bayes Accuracy is: %.3f" % NB_accuracy)
print("KNN Accuracy is: %.3f" % KNN_accuracy)

print("\nNaive Bayes Precision is: %.3f" % NB_precision)
print("KNN Precision is: %.3f" % KNN_precision)

print("\nNaive Bayes Recall is: %.3f" % NB_recall)
print("KNN Recall is: %.3f" % KNN_recall)

print("\nNaive Bayes F1_score is: %.3f" % NB_f1_score)
print("KNN F1_score is: %.3f" % KNN_f1_score)

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

# # *** KNN FROM SCRATCH ***

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

print("K=", max_accuracy_knn, "Accuracy=", "{:.2f}%".format(max_accuracy*100), end="\n\n")

# Plot the graph
ml = MultipleLocator(5)
plt.plot(K_values, Acc_values)
plt.xlabel("K Values")
plt.ylabel("Accuracy Values")
plt.title("Accuracy vs. K values")
plt.axes().xaxis.set_minor_locator(ml)
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









