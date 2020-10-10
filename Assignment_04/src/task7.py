import pandas as pd
from sklearn import svm
from sklearn.model_selection import KFold, train_test_split, cross_val_score
from sklearn import metrics

train_df = pd.read_csv("include/train.csv")
test_df = pd.read_csv("include/test.csv")
results_df = pd.read_csv("include/gender_submission.csv")

# train_df.info()
# test_df.info()

# Feature engineering

# Drop Cabin features as it seems to have too many unknown values
train_df = train_df.drop(['Cabin'], axis = 1)
test_df = test_df.drop(['Cabin'], axis = 1)
train_df = train_df.drop(['Name'], axis = 1)
test_df = test_df.drop(['Name'], axis = 1)
train_df = train_df.drop(['Ticket'], axis = 1)
test_df = test_df.drop(['Ticket'], axis = 1)

# Fill null values in the Embarked feature using mode, most common occurance.

Embarked_mode = train_df['Embarked'].mode()
train_df['Embarked'].fillna(Embarked_mode[0], inplace = True)

# Fill null values in the Age feature using mode, most common occurance.

Age_mode_train = train_df['Age'].mode()
train_df['Age'].fillna(Age_mode_train[0], inplace = True)

Age_mode_test = test_df['Age'].mode()
test_df['Age'].fillna(Age_mode_test[0], inplace = True)

# Fill the only null value of the Fare feature inn testing set with the mode
Fare_mode_test = test_df['Fare'].mode()
test_df['Fare'].fillna(Fare_mode_test[0], inplace = True)

# Map the Sex feature in the dataset to binary values: female = 1 and male = 0
train_df['Sex'] = train_df['Sex'].map({'female': 1, 'male':0})
test_df['Sex'] = test_df['Sex'].map({'female': 1, 'male':0})

# Map the Embarked feature in the datset to binary values: S = 0, C = 1 and Q = 2
train_df['Embarked'] = train_df['Embarked'].map({'S': 0, 'C':1, 'Q':2})
test_df['Embarked'] = test_df['Embarked'].map({'S': 0, 'C':1, 'Q':2})

data_features = ['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = train_df['Survived'].values
features = train_df[data_features].values

X_train = train_df[data_features].values
y_train = train_df['Survived'].values
X_test = test_df[data_features].values
y_test = results_df['Survived'].values

### K Fold Cross Validation with K = 5
cv = KFold(n_splits=5, random_state=None, shuffle=False)

# Create a linear SVM classifier

clf_linear = svm.SVC(kernel='linear')

i = 1
accuracy, precision, recall, f1_score = [], [], [], []

for train_index, test_index in cv.split(X_train):
	# Break up X_train and y_train in to test/train datasets for CV
    X_cv_train, y_cv_train = X_train[train_index], y_train[train_index]
    X_cv_test, y_cv_test = X_train[test_index], y_train[test_index]

    clf_linear.fit(X_cv_train, y_cv_train)
    y_cv_pred = clf_linear.predict(X_cv_test)

    accuracy.append(metrics.accuracy_score(y_cv_test, y_cv_pred))
    precision.append(metrics.precision_score(y_cv_test, y_cv_pred))
    recall.append(metrics.recall_score(y_cv_test, y_cv_pred))
    f1_score.append(metrics.f1_score(y_cv_test, y_cv_pred))

    i = i+1
    # print("=====================================")

print("\n\nAverage metrics over five folds for LINEAR KERNEL: ")
print("\nThe average accuracy is: %.4f" % (sum(accuracy)/len(accuracy)))
print("The average precision is: %.4f" % (sum(precision)/len(precision)))
print("The average recall is: %.4f" % (sum(recall)/len(recall)))
print("The average f1_score is: %.4f" % (sum(f1_score)/len(f1_score)))

clf_linear.fit(X_train, y_train)
y_predict = clf_linear.predict(X_test)

print("\nSVM LINEAR KERNEL METRICS:")
print("\nThe accuracy on the entire model is: %.4f" % metrics.accuracy_score(y_test, y_predict))
print("The precision on the entire model is: %.4f" % metrics.precision_score(y_test, y_predict))
print("The recall on the entire model is: %.4f" % metrics.recall_score(y_test, y_predict))
print("The f1 score on the entire model is: %.4f" % metrics.f1_score(y_test, y_predict))


# Create a Quadratic SVM classifier

clf_quadratic = svm.SVC(kernel='poly', degree = 2)

i = 1
accuracy, precision, recall, f1_score = [], [], [], []

for train_index, test_index in cv.split(X_train):
	# Break up X_train and y_train in to test/train datasets for CV
    X_cv_train, y_cv_train = X_train[train_index], y_train[train_index]
    X_cv_test, y_cv_test = X_train[test_index], y_train[test_index]

    clf_quadratic.fit(X_cv_train, y_cv_train)
    y_cv_pred = clf_quadratic.predict(X_cv_test)

    accuracy.append(metrics.accuracy_score(y_cv_test, y_cv_pred))
    precision.append(metrics.precision_score(y_cv_test, y_cv_pred))
    recall.append(metrics.recall_score(y_cv_test, y_cv_pred))
    f1_score.append(metrics.f1_score(y_cv_test, y_cv_pred))

    i = i+1
    # print("=====================================")

print("\n\nAverage metrics over five folds for QUADRATIC KERNEL: ")
print("\nThe average accuracy is: %.4f" % (sum(accuracy)/len(accuracy)))
print("The average precision is: %.4f" % (sum(precision)/len(precision)))
print("The average recall is: %.4f" % (sum(recall)/len(recall)))
print("The average f1_score is: %.4f" % (sum(f1_score)/len(f1_score)))

clf_quadratic.fit(X_train, y_train)
y_predict = clf_quadratic.predict(X_test)

print("\nSVM QUADRATIC KERNEL METRICS:")
print("\nThe accuracy on the entire model is: %.4f" % metrics.accuracy_score(y_test, y_predict))
print("The precision on the entire model is: %.4f" % metrics.precision_score(y_test, y_predict))
print("The recall on the entire model is: %.4f" % metrics.recall_score(y_test, y_predict))
print("The f1 score on the entire model is: %.4f" % metrics.f1_score(y_test, y_predict))


# Create a RBF SVM classifier

clf_rbf = svm.SVC(kernel='rbf', C = 10.0, gamma = 0.1)

i = 1
accuracy, precision, recall, f1_score = [], [], [], []

for train_index, test_index in cv.split(X_train):
	# Break up X_train and y_train in to test/train datasets for CV
    X_cv_train, y_cv_train = X_train[train_index], y_train[train_index]
    X_cv_test, y_cv_test = X_train[test_index], y_train[test_index]

    clf_rbf.fit(X_cv_train, y_cv_train)
    y_cv_pred = clf_rbf.predict(X_cv_test)

    accuracy.append(metrics.accuracy_score(y_cv_test, y_cv_pred))
    precision.append(metrics.precision_score(y_cv_test, y_cv_pred))
    recall.append(metrics.recall_score(y_cv_test, y_cv_pred))
    f1_score.append(metrics.f1_score(y_cv_test, y_cv_pred))

    i = i+1
    # print("=====================================")

print("\n\nAverage metrics over five folds for RBF KERNEL: ")
print("\nThe average accuracy is: %.4f" % (sum(accuracy)/len(accuracy)))
print("The average precision is: %.4f" % (sum(precision)/len(precision)))
print("The average recall is: %.4f" % (sum(recall)/len(recall)))
print("The average f1_score is: %.4f" % (sum(f1_score)/len(f1_score)))

clf_rbf.fit(X_train, y_train)
y_predict = clf_rbf.predict(X_test)

print("\nSVM RBF KERNEL METRICS:")
print("\nThe accuracy on the entire model is: %.4f" % metrics.accuracy_score(y_test, y_predict))
print("The precision on the entire model is: %.4f" % metrics.precision_score(y_test, y_predict))
print("The recall on the entire model is: %.4f" % metrics.recall_score(y_test, y_predict))
print("The f1 score on the entire model is: %.4f" % metrics.f1_score(y_test, y_predict))














