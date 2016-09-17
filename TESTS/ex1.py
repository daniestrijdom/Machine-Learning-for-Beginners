from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
import pandas as pd
iris = load_iris()
test_idx = [0, 50, 100]

df = pd.DataFrame(iris.data)
df['target'] = pd.Series(iris.target)


train = df.drop([0,50,100], axis=0)
test = df.iloc[[0,50,100]]


clf = tree.DecisionTreeClassifier()
clf.fit(df.drop('target',axis=1), df.target)

print test.target
print clf.predict(test.drop('target',axis=1))

'''
#vis code
from sklearn.externadf.drop('target',axis=1)ls.six import StringIO
import pydot
dot_data = StringIO()

tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)

graph = pydot.graph_from_dot_data(dot_data.getvalue())

graph.write_png("iris.svg")

print testing_data[1], testing_target[1]
print iris.feature_names, iris.target_names
'''
