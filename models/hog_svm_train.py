from sklearn import svm
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

features = np.load("features.npy")
labels = np.load("labels.npy")

test_features = np.load("test_features.npy")
test_labels = np.load("test_labels.npy")

clf = svm.SVC(probability=True, kernel="poly")

clf.fit(features, labels)

pred = clf.predict(test_features)

print("Accuracy: " + str(accuracy_score(test_labels, pred)))
print(classification_report(test_labels, pred))
