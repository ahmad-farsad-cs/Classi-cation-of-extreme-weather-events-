import numpy as np
import pandas as pd

df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")
df_train = df_train.drop(["S.No","time"], axis=1)
df_test = df_test.drop(["S.No","time"], axis=1)

trainset = df_train.to_numpy()
testset = df_test.to_numpy()
trainset = np.insert(trainset, -1, 1, axis=1)
testset = np.insert(testset, testset.shape[1], 1, axis=1)

# Min Max scalar
min = trainset[:,:-2].min(axis=0)
max = trainset[:,:-2].max(axis=0)
dist = max - min
trainset[:,:-2] = (trainset[:,:-2] - min) / dist
testset[:,:-1] = (testset[:,:-1] - min) / dist

np.savetxt("testset.csv", testset, delimiter=',')
np.savetxt("trainset.csv", trainset, delimiter=',')
print("Preprocessed dataset output.")