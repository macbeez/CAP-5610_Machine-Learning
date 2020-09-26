import pandas as pd
from sklearn import preprocessing
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import nltk # needed for Naive-Bayes
import numpy as np

train_df = pd.read_csv("include/train2.csv")
test_df = pd.read_csv("include/test2.csv")

# train_df.info()
# test_df.info()

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

print("\nNaive Bayes Accuracy is: %.4f" % NB_accuracy)
print("KNN Accuracy is: %.4f" % KNN_accuracy)

print("\nNaive Bayes Precision is: %.4f" % NB_precision)
print("KNN Precision is: %.4f" % KNN_precision)

print("\nNaive Bayes Recall is: %.4f" % NB_recall)
print("KNN Recall is: %.4f" % KNN_recall)

print("\nNaive Bayes F1_score is: %.4f" % NB_f1_score)
print("KNN F1_score is: %.4f" % KNN_f1_score)

