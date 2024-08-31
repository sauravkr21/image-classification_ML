import pickle

from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier


# SVM classifier
svm1 = svm.SVC()
svm2 = svm.SVC()

# KNN classifier
knn1 = KNeighborsClassifier(n_neighbors=10)
knn2 = KNeighborsClassifier(n_neighbors=10)

# Logistic Regression classifier
logistic1 = LogisticRegression()
logistic2 = LogisticRegression()

# Decision Tree classifier
tree1 = DecisionTreeClassifier()
tree2 = DecisionTreeClassifier()

# load the faeture vector and labels
with open('features.pkl', 'rb') as f:
    fvec = pickle.load(f)
with open('augmented_lbl.pkl', 'rb') as f:
    lbl = pickle.load(f)

# train the classifiers 1 and 2 on augmented and original data respectively
svm1.fit(fvec, lbl)
svm2.fit(fvec[0:50000], lbl[0:50000])
print('SVM trained')

knn1.fit(fvec, lbl)
knn2.fit(fvec[0:50000], lbl[0:50000])
print('KNN trained')

logistic1.fit(fvec, lbl)
logistic2.fit(fvec[0:50000], lbl[0:50000])
print('Logistic Regression trained')

tree1.fit(fvec, lbl)
tree2.fit(fvec[0:50000], lbl[0:50000])
print('Decision Tree trained')

# load the test data
with open('test_vectors.pkl', 'rb') as f:
    test_fvec = pickle.load(f)

# predict the labels for the test data
svm1_pred = svm1.predict(test_fvec)
svm2_pred = svm2.predict(test_fvec)

knn1_pred = knn1.predict(test_fvec)
knn2_pred = knn2.predict(test_fvec)

logistic1_pred = logistic1.predict(test_fvec)
logistic2_pred = logistic2.predict(test_fvec)

tree1_pred = tree1.predict(test_fvec)
tree2_pred = tree2.predict(test_fvec)

# save the predictions

with open('svm1_pred.pkl', 'wb') as f:
    pickle.dump(svm1_pred, f)

with open('svm2_pred.pkl', 'wb') as f:
    pickle.dump(svm2_pred, f)

with open('knn1_pred.pkl', 'wb') as f:
    pickle.dump(knn1_pred, f)

with open('knn2_pred.pkl', 'wb') as f:
    pickle.dump(knn2_pred, f)

with open('logistic1_pred.pkl', 'wb') as f:
    pickle.dump(logistic1_pred, f)

with open('logistic2_pred.pkl', 'wb') as f:
    pickle.dump(logistic2_pred, f)

with open('tree1_pred.pkl', 'wb') as f:
    pickle.dump(tree1_pred, f)

with open('tree2_pred.pkl', 'wb') as f:
    pickle.dump(tree2_pred, f)
