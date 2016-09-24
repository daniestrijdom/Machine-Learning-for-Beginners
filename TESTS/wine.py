'''
Linear regression on red wine quality
'''
import pandas as pd

# NOTE: 1 = red, 0 = white
df1 = pd.read_csv("winequality-red.csv", delimiter=";")
df1['type'] = 1
df2 = pd.read_csv("winequality-white.csv", delimiter=";")
df2['type'] = 0

df = df1.append(df2)

train_len = int(round(len(df)*0.8,0))
test_len = len(df) - train_len

# TODO: Specify testing and trainig sets
train_idx = [i for i in range(train_len,len(df))]
train_set = df.drop(train_idx, axis=0)

test_idx = [ i for i in range(0,train_len)]
test_set = df.drop(train_idx, axis=0)
test_set = test_set.reset_index(drop=True)

# TODO: Fit reg modeln - Decision Tree
from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf.fit(train_set.drop('type',axis=1),train_set.type)

result1 = clf.predict(test_set.drop('type',axis=1))

# TODO: Cross validation
def cross_validate(clf, df):
    from sklearn import cross_validation
    n_samples = len(df)

    cv = cross_validation.ShuffleSplit(n_samples, n_iter=5, test_size=0.2, random_state=0)
    c_valid = cross_validation.cross_val_score(clf, df.drop('type', axis=1), df.type, cv=cv)
    return sum(c_valid)/len(c_valid)

# TODO: Fit reg modeln - Decision Tree
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
result2 = gnb.fit(train_set.drop('type',axis=1),train_set.type).predict(test_set.drop('type',axis=1))


# NOTE: OUtput
from sklearn.metrics import accuracy_score
import math

print '''
----- Wine Type (red/white) -----
\nClassifier: Decision tree
Accuracy Score: %f
Cross Validation: %f
\n
''' % (accuracy_score(list(test_set['type']),result1), cross_validate(clf, df))

print '''
----- Wine Type (red/white) -----
\nClassifier: Gaussian Naive Bayes
Accuracy Score: %f
Cross Validation: %f
\n
''' % (accuracy_score(list(test_set['type']),result2), cross_validate(gnb, df))

# TODO: Plot this motherfucker
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print df.columns
df[df['type']== 1].plot.scatter(x='chlorides',y='pH', marker=".")
df[df['type']== 0].plot.scatter(x='chlorides',y='pH', marker="o")
plt.show()
