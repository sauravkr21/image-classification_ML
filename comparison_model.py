# Accuracy of the models

import pickle
from sklearn.metrics import accuracy_score

# load the test labels
with open('test_lbl.pkl', 'rb') as f:
    test_lbl = pickle.load(f)

# load the predictions
with open('svm1_pred.pkl', 'rb') as f:
    svm1_pred = pickle.load(f)

with open('svm2_pred.pkl', 'rb') as f:
    svm2_pred = pickle.load(f)

with open('knn1_pred.pkl', 'rb') as f:
    knn1_pred = pickle.load(f)

with open('knn2_pred.pkl', 'rb') as f:
    knn2_pred = pickle.load(f)

with open('logistic1_pred.pkl', 'rb') as f:
    logistic1_pred = pickle.load(f)

with open('logistic2_pred.pkl', 'rb') as f:
    logistic2_pred = pickle.load(f)

with open('tree1_pred.pkl', 'rb') as f:
    tree1_pred = pickle.load(f)

with open('tree2_pred.pkl', 'rb') as f:
    tree2_pred = pickle.load(f)

with open('predictions.pkl', 'rb') as f:
    pred1 = pickle.load(f)

with open('predictions2.pkl', 'rb') as f:
    pred2 = pickle.load(f)

# calculate the accuracy of the models
svm1_acc = accuracy_score(test_lbl, svm1_pred)
svm2_acc = accuracy_score(test_lbl, svm2_pred)

knn1_acc = accuracy_score(test_lbl, knn1_pred)
knn2_acc = accuracy_score(test_lbl, knn2_pred)

logistic1_acc = accuracy_score(test_lbl, logistic1_pred)
logistic2_acc = accuracy_score(test_lbl, logistic2_pred)

tree1_acc = accuracy_score(test_lbl, tree1_pred)
tree2_acc = accuracy_score(test_lbl, tree2_pred)

pred1_acc = accuracy_score(test_lbl, pred1)
pred2_acc = accuracy_score(test_lbl, pred2)

print('SVM1 accuracy: ', svm1_acc)
print('SVM2 accuracy: ', svm2_acc)
print('KNN1 accuracy: ', knn1_acc)
print('KNN2 accuracy: ', knn2_acc)
print('Logistic1 accuracy: ', logistic1_acc)
print('Logistic2 accuracy: ', logistic2_acc)
print('Tree1 accuracy: ', tree1_acc)
print('Tree2 accuracy: ', tree2_acc)
print('MLP1 accuracy: ', pred1_acc)
print('MLP2 accuracy: ', pred2_acc)
