from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report

from sklearn.decomposition import PCA
from sklearn.svm import SVC




lfw_people = fetch_lfw_people(min_faces_per_person=200, resize=0.4)
X = lfw_people.data
y = lfw_people.target
target_names = lfw_people.target_names

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=100)


n_components = 150

pca = PCA(n_components=n_components, svd_solver='randomized',
          whiten=True).fit(X_train)


X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)



print("Fitting the classifier to the training set")

param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
              'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
clf = GridSearchCV(
    SVC(class_weight='balanced'), param_grid
)
clf = clf.fit(X_train_pca, y_train)

print("Best estimator:",end=" ")
print(clf.best_estimator_)


print("Predicting people's names on the test set")
y_pred = clf.predict(X_test_pca)
print(classification_report(y_test, y_pred, target_names=target_names))

clf_neural = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
clf_neural = clf_neural.fit(X_train_pca, y_train)
y_pred = clf_neural.predict(X_test_pca)

print(classification_report(y_test, y_pred, target_names=target_names))


