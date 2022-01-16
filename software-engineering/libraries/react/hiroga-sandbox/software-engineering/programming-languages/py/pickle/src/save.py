from sklearn import datasets
iris = datasets.load_iris()
X, y = iris.data[:-1], iris.target[:-1]

# K近傍法(最近傍法)
from sklearn.neighbors import KNeighborsClassifier

clf = KNeighborsClassifier()
clf.fit(X, y)

import pickle
with open('/src/clf/model.pickle', mode='wb') as fp:
    pickle.dump(clf, fp)
