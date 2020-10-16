import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from progress.bar import Bar
# import matplotlib.pyplot.savefig
import matplotlib.pyplot as plt
from tabulate import tabulate

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")

target = train_df['label'].values
features = train_df.drop(['label'], axis = 1)

############################################################################################################
# DATA ANALYSIS
############################################################################################################

# train_df.info()
# test_df.info()

# print(train_df.head())
# print(train_df.isnull().sum())

# # Plot the labels (digits) in the training dataset

# digits = train_df['label']
# # print(digits)

# freq = {}
# for digit in digits:
#     if (digit in freq):
#         freq[digit] += 1
#     else:
#         freq[digit] = 1

# data = []
# keys = sorted(list(freq.keys()))

# for key in keys:
# 	data.append([key, freq[key]])

# print(tabulate(data, headers = ["Digit", "Count"], tablefmt = 'psql'))

# label = list(freq.keys())
# counts = list(freq.values())

# # print("labels and the length of label list: \n", label, len(label))
# # print("labels and the length of counts list: \n", counts, len(counts))

# fig = plt.figure(figsize = (10,7))
# plt.pie(counts, labels = label, autopct='%1.1f%%')

# plt.savefig("digit_count.png")

#############################################################################################################
# FINDING THE BEST K VALUES 
#############################################################################################################

# k = 1
# acc = []
# max_k = 100
# bar = Bar('Processing', max = max_k)

# while(k <= max_k):
#   # Build training and testing dataset
#   X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 520)
#   # Create a KNN Classifier
#   knn = KNeighborsClassifier(n_neighbors = k)
#   # Train the model using the training sets
#   knn.fit(X_train, y_train)
#   # Predict the response for the test_titanic dataset
#   y_pred = knn.predict(X_test)

#   acc.append([k, metrics.accuracy_score(y_test, y_pred)])
#   k = k + 1
#   bar.next()

# bar.finish()

# K_values = [row[0] for row in acc]
# Acc_values = [row[1] for row in acc]

# max_accuracy = max(Acc_values)
# max_accuracy_knn = K_values[Acc_values.index(max_accuracy)]

# print("When K=", max_accuracy_knn, ", the maximum accuracy is obtained. Accuracy =", "{:.2f}%".format(max_accuracy*100), end="\n\n")

############################################################################################################
# KNN ON THE TRAINING DATA USING K VALUE THAT GIVES MAXIMUM ACCURACY
############################################################################################################

X_train, X_test, y_train, y_test = train_test_split(features, target, test_size = 0.3, random_state = 600)

# Create a KNN Classifier
knn = KNeighborsClassifier(n_neighbors = 5)

# Train the model using the training sets
knn.fit(X_train, y_train)

# Predict the response for the test_titanic dataset
y_pred = knn.predict(X_test)
# print("KNN: Predicted y labels: ", y_pred)

print("\nThe accuracy on the entire model is: %.4f" % metrics.accuracy_score(y_test, y_pred))

############################################################################################################
# WRITE K AND ACCURACY VALUES INTO A TEXT FILE
############################################################################################################

# with open('K_and_Acc.csv', 'a') as a_writer:
#   for k_val, accu in zip(K_values, Acc_values):
#     a_writer.write(str(k_val) + "," + str(accu) + ";\n")

############################################################################################################
# PLOT K VALUES vs. ACCURACY
############################################################################################################

# plt.plot(K_values, Acc_values)
# plt.xlabel("K Values")
# plt.ylabel("Accuracy Values")
# plt.title("K values vs. Accuracy")
# plt.savefig("K_vs_Acc.png")











