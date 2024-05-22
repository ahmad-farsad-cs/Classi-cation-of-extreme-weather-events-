from sklearn.svm import SVC
import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train = df_train.drop(["S.No","time"], axis=1)
df_test = df_test.drop(["S.No","time"], axis=1)
trainset = df_train.to_numpy()
testset = df_test.to_numpy()

# Min Max scalar
min = trainset[:,:-1].min(axis=0)
max = trainset[:,:-1].max(axis=0)
dist = max - min
trainset[:,:-1] = (trainset[:,:-1] - min) / dist
testset= (testset - min) / dist

acc_mean = []
for i in range(5):
    model = SVC(kernel="poly")
    np.random.shuffle(trainset)
    n = len(trainset)//10
    model.fit(trainset[n:,:-1], trainset[n:,-1])
    res = model.predict(trainset[:n,:-1])
    acc_mean.append(np.mean(res == trainset[:n,-1]))
print(sum(acc_mean)/len(acc_mean))