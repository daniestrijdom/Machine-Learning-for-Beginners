'''
Linear regression on red wine quality
- PCA
'''

# TODO: get data
import pandas as pd

df = pd.read_csv("winequality-red.csv", delimiter=";")

def do_pca(data):
    from sklearn.decomposition import PCA

    pca = PCA(n_components=len(data.columns))

    return pca.fit_transform(data)

df = pd.DataFrame(do_pca(df))
# TODO: Specify testing and trainig sets

train_idx = [i for i in range(1550,1599)]
train = df.drop(train_idx, axis=0)

test_idx = [ i for i in range(0,1550)]
test = df.drop(test_idx, axis=0)
test = test.reset_index(drop=True)

# TODO: Fit reg model
from sklearn import linear_model

clf = linear_model.LinearRegression()
clf.fit(train.drop(11,axis=1),train[11])

# TODO: Assess prediction ability

result = clf.predict(test.drop(11,axis=1))
