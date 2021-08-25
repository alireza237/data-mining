from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier


def algorithm_decision(x_train, y_train, x_test, y_test, algorithm, name):
    classifier = DecisionTreeClassifier(criterion=algorithm)
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    print("-----------{} Decision tree: -----------------".format(name))
    print("+++++++++++ confusion matrix +++++++++++++++++++")
    print(confusion_matrix(y_test, y_prediction))
    print("+++++++++++ classification result ++++++++++++++")
    print(classification_report(y_test, y_prediction))
    correct = 0
    for i in range(len(y_prediction)):
        if y_test[i] == y_prediction[i]:
            correct += 1
    print("+++++++++++ Accuracy ++++++++++++++++++++")
    print(correct / float(len(y_test)))

    metrics.plot_roc_curve(classifier, x_test, y_test)
    plt.show()


def Random(x_train, y_train, x_test, y_test, name):
    classifier = RandomForestClassifier()
    classifier.fit(x_train, y_train)
    y_prediction = classifier.predict(x_test)
    print("-----------{} Decision tree: -----------------".format(name))
    print("+++++++++++ confusion matrix +++++++++++++++++++")
    print(confusion_matrix(y_test, y_prediction))
    print("+++++++++++ classification result ++++++++++++++")
    print(classification_report(y_test, y_prediction))
    correct = 0
    for i in range(len(y_prediction)):
        if y_test[i] == y_prediction[i]:
            correct += 1
    print("+++++++++++ Accuracy ++++++++++++++++++++")
    print(correct / float(len(y_test)))

    metrics.plot_roc_curve(classifier, x_test, y_test)
    plt.show()


traindata = csv.reader(open('diabetes.csv'))
train = []
for row in traindata:
    train.append(list(row))

y = []
x = []

for i in range(1, len(train)):
    y.append((train[i][8]))

for j in range(1, len(train)):
    b = []
    for k in range(8):
        b.append((train[j][k]))

    x.append(b)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)




algorithm_decision(x_train, y_train, x_test, y_test, "entropy", "C4.5")
print("\n\n\n")
Random(x_train, y_train, x_test, y_test, "RandomForest")
