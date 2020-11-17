import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn import metrics
import matplotlib.pyplot as plt
from tabulate import tabulate

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")
output_df = pd.read_csv("include/output.csv") # Has 100% accuracy on testing data.

X_train = train_df.drop(['label'], axis = 1).values # Drop target and get values of features
y_train = train_df['label'].values
X_test = test_df.values
y_test = [] # ground truth not given
y_output = output_df['Label'].values

#############################################################################################################
# FIVE FOLD CROSS VALIDATION
#############################################################################################################

# K Fold Cross Validation with K = 5
cv = KFold(n_splits = 5, random_state = None, shuffle = False)

# create a KNN classifier by choosing the best K value obtained, K = 5
knn = KNeighborsClassifier(n_neighbors = 1)

i = 1
accuracy, precision, recall, f1_score = [], [], [], []
final_y_pred = np.array([])

for train_index, test_index in cv.split(X_train):
    print("\n\n*** FOLD ", i, " ***")
    print("\nTraining_index: ", train_index)
    print("Testing_index: ", test_index)

    X_cv_train, y_cv_train = X_train[train_index], y_train[train_index]
    X_cv_test, y_cv_test = X_train[test_index], y_train[test_index]

    print("Size of the training data: ", len(train_index))
    print("Size of the testing data: ", len(test_index))

    knn.fit(X_cv_train, y_cv_train)
    y_cv_pred = knn.predict(X_cv_test)
    # print("y_test size: ", len(y_cv_test))
    # print("y_pred size: ", len(y_cv_pred))

    # print("Iteration number and y_pred: ", i, y_cv_pred)

    accuracy.append(metrics.accuracy_score(y_cv_test, y_cv_pred))
    precision.append(metrics.precision_score(y_cv_test, y_cv_pred, average = None))
    recall.append(metrics.recall_score(y_cv_test, y_cv_pred, average = None))
    f1_score.append(metrics.f1_score(y_cv_test, y_cv_pred, average = None))

    final_y_pred = np.concatenate((final_y_pred, y_cv_pred), axis = None)
    # print("\nLength of final_y_pred: ", len(final_y_pred))
    # print("final_y_pred: ", final_y_pred)
    i = i + 1

# averaging over all the values for each fold:
for i, val in enumerate(precision):
    precision[i] = precision[i].mean()

for i, val in enumerate(recall):
    recall[i] = recall[i].mean()

for i, val in enumerate(f1_score):
    f1_score[i] = f1_score[i].mean()

# print("\nACCURACY: ", accuracy)
# print("PRECISION: ", precision)
# print("RECALL: ", recall)
# print("F1 SCORE: ", f1_score)

table_print = []
for acc, pre, rec, f1 in zip(accuracy, precision, recall, f1_score):
    table_print.append([acc,pre,rec,f1])

print("\nKNN Evaluation Metrics:")
print(tabulate(table_print, headers = ["Accuracy", "Precision", "Recall", "F1 Score"], tablefmt = 'psql'))

print("Average metrics over five folds: ")
print("\nThe average accuracy is: %.4f" % (sum(accuracy)/len(accuracy)))
print("The average precision is: %.4f" % (sum(precision)/len(precision)))
print("The average recall is: %.4f" % (sum(recall)/len(recall)))
print("The average f1_score is: %.4f" % (sum(f1_score)/len(f1_score)))

###############################################################################################
# Fit and predict on the entire model and find the accuracy
###############################################################################################

knn.fit(X_train, y_train)
y_predict = knn.predict(X_test)
# print("Y_prediction: ", y_predict)

test_accuracy = metrics.accuracy_score(y_output, y_predict)
test_accuracy = test_accuracy * 100
print("\nAccuracy of the testing dataset: %.4f" % test_accuracy, "%")

###############################################################################################
# Write Y_prediction values into a csv
###############################################################################################

ImageId = [x for x in range(len(y_predict))]
with open('sample_submission.csv', 'a') as a_writer:
  a_writer.write("ImageId" + "," + "label" + "\n")  
  for img_id, y_pred in zip(ImageId, y_predict):
    a_writer.write(str(img_id) + "," + str(y_pred) + "\n")

#########################################################################################
# Plot Pie chart
#########################################################################################

# Plot the labels (digits) in the training dataset

digits = y_train
# print(digits)

freq = {}
for digit in digits:
    if (digit in freq):
        freq[digit] += 1
    else:
        freq[digit] = 1

data = []
keys = sorted(list(freq.keys()))

for key in keys:
  data.append([key, freq[key]])

print("Actual Digit Frequencies")
print(tabulate(data, headers = ["Digit", "Count"], tablefmt = 'psql'))

label_train = list(freq.keys())
counts_train = list(freq.values())

# print("labels and the length of label list: \n", label_train, len(label_train))
# print("labels and the length of counts list: \n", counts_train, len(counts_train))

fig = plt.figure(figsize = (10,7))

plt.pie(counts_train, labels = label_train, autopct='%1.1f%%')
plt.title("\nActual digit frequencies")
plt.savefig("digit_count_ground_truth.png")

# Plot the labels (digits) that are predicted 

digits = final_y_pred
# print(digits)

freq = {}
for digit in digits:
    if (digit in freq):
        freq[digit] += 1
    else:
        freq[digit] = 1

data = []
keys = sorted(list(freq.keys()))

for key in keys:
  data.append([key, freq[key]])

print("Predicted Digit Frequencies")
print(tabulate(data, headers = ["Digit", "Count"], tablefmt = 'psql'))

label_pred = list(freq.keys())
counts_pred = list(freq.values())

# print("labels and the length of label list: \n", label_pred, len(label_pred))
# print("labels and the length of counts list: \n", counts_pred, len(counts_pred))

fig = plt.figure(figsize = (10,7))
plt.pie(counts_pred, labels = label_pred, autopct='%1.1f%%')
plt.title("Predicted digit frequencies")
plt.savefig("digit_count_predicted.png")




