#making moduler comparison

import prep_terrain_data

feature_train, label_train, feature_test, label_test = prep_terrain_data.makeTerrainData()

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def makingClassifier (method, k=3): #method is the classifier type, k for the K Neighbours parameter (not necessary)
	if int(method) == 1:
		clf = GaussianNB()
	elif int(method) == 2:
		clf = svm.SVC()
	elif int(method) == 3:
		clf = tree.DecisionTreeClassifier()
	elif int(method) == 4:
		clf = KNeighborsClassifier(n_neighbors = k)
	
	clf.fit(feature_train, label_train)
	pred= clf.predict (feature_test)
	acc = accuracy_score(pred, label_test)
	print ("Accuracy Score for ", classifiers[method-1], " is ", acc, "\n")
	return acc

def addingToPlot (acc, method, i, x_tick_num, x_tick_name):
	x = [[i]]
	y = [acc]
	colors = ["green", "blue", "yellow", "red", "purple"]
	plt.scatter (x, y, s=30, color = colors[i-1])
	x_tick_num.append([i])
	x_tick_name.append(classifiers[method-1])
	plt.xticks(x_tick_num, x_tick_name)
	#print (x, y, method, i, x_tick_name, x_tick_num)


#Main
i = 1
x_tick_num = []
x_tick_name = []
accuracy_list = []
classifiers = ["Naive Bayes", "SVM", "Decision Tree", "K Neighbours"]
while (True):
	print ("Please type the number which classifier do you want to use: 1 for Naive Bayes, 2 for SVM, 3 for Decision Tree, 4 for K Neighbours, 5 for exit")
	print ("Type 5 to exit to stop and see the graph")
	methods = int(input())
	
	if methods == 5:
	    break

	acc_result = makingClassifier(methods)
	accuracy_list.append (acc_result)
	
	addingToPlot(acc_result, methods, i, x_tick_num, x_tick_name)
	i = i+1

plt.xlabel("different classifiers")
plt.ylabel("accuracy scores")
plt.title("Comparison of Classifiers")

print ("Here is the graph of the comparison")
plt.show()

max_acc = max(accuracy_list)
index_acc = accuracy_list.index(max_acc)
print ("Best classifier is ", x_tick_name[index_acc], " with a accuracy score of ", max_acc, "\n")

print ("Press any key to exit!")
t = input()