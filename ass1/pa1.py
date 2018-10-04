import csv
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score


# Linear Regression class
class LinearRegression:
    def __init__(self, datas, result):
        self.x = datas
        self.y = result
        self.thetas = initTheta(len(datas[0]))
        # number of datas
        self.dataNum = len(datass)
        # number of Xi(feature)
        self.ftNum = len(self.thetas)

    def initTheta(self, nums):
        rt = []
        for i in range(0, nums):
            rt.append(np.random.random())
        return rt

    # Cost Function
    def costFunc(self):
        J = 0
        for i in range(0, self.dataNum):
            tmp = 0
            for j in range(0, self.ftNum):
                tmp = tmp + self.thetas[j] * self.x[i][j]
            J += (tmp - self.y[i])**2
        return J/2

    # idx : for specific X value
    def wgtVals(self, idx):
        rt = 0
        # Sum up all the values
        for i in range(0, self.dataNum):
            # compute (X^T*theta)
            tmp = 0
            for j in range(0, self.ftNum):
                tmp = tmp + self.thetas[j] * self.x[i][j]
            rt += (tmp - self.y[i]) * self.x[i][idx]
        return rt

    def gradientDescent(self, alpha=10**-1, ep=0.0001, max_iter=10000):

        count = 0
        converged = False
        # J
        J = costFunc()

        while not converged:
            idx = 0
            # Get thetas
            for j in range(0, self.ftNum):
                # Do we need to devide the number of datas
                thetas[j] = thetas[j] - alpha * self.wgtVals(j)

            e = costFunc()

            if abs(J-e) <= ep:
                print 'Converged, iterations: ', count, '!!!'
                converged = True
            J = e
            count += 1

            if count == max_iter:
                print 'Max interactions exceeded!'
                converged = True


def importCsv(path, delimiter, isHead=True):
    x, y = [], []
    ftrNum = 0

    with open(path) as f:
        lines = f.readlines()

        for line in lines:

            if isHead:
                isHead = False
                continue

            arr = line.split(delimiter)
            y.append(arr.pop().replace("\n", ""))
            x.append(arr)

    return [x, y]


"""open csv"""
# devData = pd.read_csv('PA1_dev.csv', header = None)
# tstData = pd.read_csv('PA1_test.csv', header = None)
# trnData = pd.read_csv('PA1_train.csv', header = None)

importCsv("PA1_test.csv", ",")
print np.random.random()
# reader = csv.reader(open("PA1_test.csv", "rb"), delimiter=",")
# result = np.array(list(reader))
# print len(result[0])
# print result[1:]

# theta = gradient_descent(alpha, x, y, ep, max_iter=1000)
