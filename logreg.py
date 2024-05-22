import numpy as np
from scipy.special import softmax
import math
import pandas as pd
from prep import *

# One-hot encoder
class OneHotEncoder:
    def __init__(self):
        self.no_type = None
    
    def fit_transform(self, line):
        line = line.astype(int)
        self.no_type = len(np.unique(line))

        array = np.zeros((len(line), self.no_type))
        for i, n in enumerate(line):
            array[i][n] = 1

        return array
    
    def transform(self, line):
        if self.no_type == None:
            return None
            
        line = line.astype(int)
        array = np.zeros((len(line), self.no_type))
        for i, n in enumerate(line):
            array[i][n] = 1

        return array

onehot_encoder = OneHotEncoder()

# Multiclass logistic regression classifier
class LogRegMultiClass:
    def __init__(self, W, lr = 0.12, reg = 0.000001):
        self.W = W
        self.lr = lr
        self.reg = reg

    def loss(self, X, Y):
        Z = - X @ self.W
        N = X.shape[0]
        loss = 1/N * (np.trace(X @ self.W @ Y.T) + np.sum(np.log(np.sum(np.exp(Z), axis=1))))
        return loss
        
    def gradient(self, X, Y):
        Z = - X @ self.W
        P = softmax(Z, axis=1)
        N = X.shape[0]
        gd = 1/N * (X.T @ (Y - P)) + 2 * self.reg * self.W
        return gd

    def gradient_descent(self, X, Y):
        self.W -= self.lr * self.gradient(X, Y)
    
    def train(self, early_stop = 400, batch_size = 32, max_iter = 5000):
        self.divide_dataset(0.1)
        n = math.ceil(self.trainset.shape[0]/batch_size)
        self.max_acc = -1
        self.best_W = self.W
        es = 0
        for epoch in range(max_iter):
            # Size 32 random mini batch
            for i in range(n):
                indice = (np.random.rand(32) * self.trainset.shape[0]).astype(int)
                sample = self.trainset[indice]
                X = sample[:, :-3]
                Y = sample[:, -3:]
                self.gradient_descent(X, Y)

            # Validation
            valid_pred = model.predict(self.validset[:,:-3])
            acc = np.mean(valid_pred == np.argmax(self.validset[:,-3:], axis=1))
            if acc > self.max_acc:
                self.max_acc = acc
                self.best_W = self.W
                es = 0
            
            # Early stop
            es += 1
            if es == early_stop:
                break
        print("Training complete, accuracy: " + str(self.max_acc))
        self.W = self.best_W
    
    def predict(self, X):
        Z = - X @ self.W
        P = softmax(Z, axis=1)
        return np.argmax(P, axis=1)
    
    def set_dataset(self, dataset):
        self.dataset = dataset
    
    def divide_dataset(self, valid_ratio):
        np.random.shuffle(self.dataset)
        n = math.ceil(self.dataset.shape[0] * valid_ratio)
        self.trainset = self.dataset[:n,:]
        self.validset = self.dataset[n:,:]


# Change dataset label form to one-hot
trainset = np.loadtxt("trainset.csv", delimiter = ",")
trainsetX = trainset[:, :-1]
trainsetY = trainset[:, -1]
trainsetY = onehot_encoder.fit_transform(trainsetY)
trainset = np.hstack((trainsetX, trainsetY))
testset = np.loadtxt("testset.csv", delimiter = ",")

# Train 3 models and assemble
res = np.zeros((testset.shape[0],3))
# acc = []
for i in range(5):
    print("Training model " + str(i+1) + " ...")
    model = LogRegMultiClass(np.random.rand(19, 3))
    model.set_dataset(trainset)
    model.train()
    # acc.append(model.max_acc)
    res += onehot_encoder.transform(model.predict(testset))

res = np.argmax(res, axis=1)

df_pred = pd.DataFrame(res, columns = ["LABELS"])
df_pred["S.No"] = range(res.shape[0])
df_pred = df_pred[["S.No", "LABELS"]]
df_pred.to_csv("pred.csv", index=False)

# print("LR: 0.11, REG: 0.0000001, LOSS: " + str(sum(acc)/len(acc)))