'''
Linear regression on red wine quality
'''
# TODO: get data
import pandas as pd

df = pd.read_csv("winequality-red.csv", delimiter=";")

# TODO: Specify testing and trainig sets

train_idx = [i for i in range(1550,1599)]
train = df.drop(train_idx, axis=0)


test_idx = [ i for i in range(0,1550)]
test = df.drop(test_idx, axis=0)
test = test.reset_index(drop=True)

# TODO: Fit reg model
from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit(train.drop("quality",axis=1),train.quality)

# print train.head()

predict = list(pd.to_numeric(pd.Series(clf.predict(test.drop('quality',axis=1)))))
e = 0
for i in range(len(predict)):
    predict[i] = int(round(predict[i],0))

actual = list(test.quality)

from sklearn.metrics import accuracy_score

print accuracy_score(actual, predict)
