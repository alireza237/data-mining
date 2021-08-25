import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.preprocessing import StandardScaler





adults = pd.read_csv('adult.csv',names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])
adults_test = pd.read_csv('adult.csv',names=['Age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','label'])


train_data = adults.drop('label',axis=1)

test_data = adults_test.drop('label',axis=1)

data = train_data.append(test_data)

label = adults['label'].append(adults_test['label'])

data.head()
full_dataset = adults.append(adults_test)

label.head()

data_binary = pd.get_dummies(data)

data_binary.head()

x_train, x_test, y_train, y_test = train_test_split(data_binary,label)

performance = []

# Gaussian Naive Bayes

GNB = GaussianNB()

 # Binary data
GNB.fit(x_train,y_train)
train_score = GNB.score(x_train,y_train)
test_score = GNB.score(x_test,y_test)
print(f'Gaussian Naive Bayes : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'Gaussian Naive Bayes', 'training_score':train_score, 'testing_score':test_score})

logClassifier = LogisticRegression()
logClassifier.fit(x_train,y_train)
train_score = logClassifier.score(x_train,y_train)
test_score = logClassifier.score(x_test,y_test)

print(f'LogisticRegression : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'LogisticRegression', 'training_score':train_score, 'testing_score':test_score})

knn_scores = []
train_scores = []
test_scores = []

for n in range(1,10,2):
    knn = KNeighborsClassifier(n_neighbors=n)
    knn.fit(x_train,y_train)
    train_score = knn.score(x_train,y_train)
    test_score = knn.score(x_test,y_test)
    train_scores.append(train_score)
    test_scores.append(test_score)
    print(f'KNN : Training score - {train_score} -- Test score - {test_score}')
    knn_scores.append({'algorithm':'KNN', 'training_score':train_score})
    
plt.scatter(x=range(1, 10, 2),y=train_scores,c='b')
plt.scatter(x=range(1, 10, 2),y=test_scores,c='r')

plt.show()

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)

knn.score(x_train,y_train)

train_score = knn.score(x_train,y_train)
test_score = knn.score(x_test,y_test)

print(f'K Neighbors : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'K Neighbors', 'training_score':train_score, 'testing_score':test_score})

rndTree = RandomForestClassifier()
rndTree.fit(x_train,y_train)
rndTree.score(x_test,y_test)
rndTree.score(x_train,y_train)

train_score = rndTree.score(x_train,y_train)
test_score = rndTree.score(x_test,y_test)

print(f'Random Forests : Training score - {train_score} - Test score - {test_score}')

performance.append({'algorithm':'Random Forests', 'training_score':train_score, 'testing_score':test_score})
svc = svm.SVC(kernel='linear')

scaler = StandardScaler()

scaler.fit(data_binary,label)
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)
svc.fit(x_train_scaled,y_train)
svc.score(x_test_scaled,y_test)
