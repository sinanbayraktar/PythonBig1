#using random data generator making my own code

import prep_terrain_data

feature_train, label_train, feature_test, label_test = prep_terrain_data.makeTerrainData()


from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import GaussianNB
clf_naive = GaussianNB()
clf_naive.fit(feature_train, label_train)
pred_naive = clf_naive.predict (feature_test)
acc_naive = accuracy_score(pred_naive, label_test)
print ("Accuracy Score for Naive Bayes is ", acc_naive)

from sklearn import svm
clf_svm = svm.SVC()
clf_svm.fit(feature_train, label_train)
pred_svm = clf_svm.predict (feature_test)
acc_svm = accuracy_score(pred_svm, label_test)
print ("Accuracy Score for SVM is ", acc_svm)

from sklearn import tree
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(feature_train, label_train)
pred_tree = clf_tree.predict(feature_test)
acc_tree = accuracy_score(pred_tree, label_test)
print ("Accuracy Score for Decision Tree is ", acc_tree)

from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(feature_train, label_train) 
pred_neigh = neigh.predict(feature_test)
acc_neigh = accuracy_score(pred_neigh, label_test)
print ("Accuracy Score for Decision Tree is ", acc_neigh)

print ("Here is the graph of the comparison")
import matplotlib.pyplot as plt
x = [[1], [2], [3], [4]]
y = [acc_naive, acc_svm, acc_tree, acc_neigh]
plt.scatter (x, y, s=30)
plt.xlabel("different classifiers")
plt.ylabel("accuracy scores")
plt.title("Comparison of Classifiers")
plt.xticks([1, 2, 3, 4], ["Naive Bayes", "SVM", "Decision Tree", "K Neighbours"])
plt.show()