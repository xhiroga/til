import pickle
with open('/src/clf/model.pickle', mode='rb') as fp:
    clf = pickle.load(fp)


from sklearn import datasets
iris = datasets.load_iris()
X_last, y_last = iris.data[-1], iris.target[-1]

print(clf.predict([X_last]))
print(y_last)
