# date to YYYYMMDD
#

import csv
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from datetime import datetime


# Linear Regression class
class LinearRegression:
    def __init__(self, datas, result):
        self.x = datas
        self.y = result
        # number of datas
        self.dataNum = len(datas)
        # number of Xi(feature)
        self.ftNum = len(self.x[0])
        self.thetas = []

    def initTheta(self, nums):
        rt = []
        for i in range(0, nums):
            # rt.append(1)
            rt.append(np.random.random())
        return rt

    # Cost Function
    def costFunc(self):
        J = 0
        for i in range(0, self.dataNum): 
            tmp = 0.0
            for j in range(0, self.ftNum):
                tmp = tmp + self.thetas[j] * self.x[i][j]

            J += (tmp - self.y[i])**2
        return (1/2)*J

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

    def gradientDescent(self, alpha=10**-1, ep=0.001, max_iter=1000):

        count = 0
        converged = False
        # self.thetas = self.initTheta(self.ftNum+1)
        self.thetas = np.zeros(self.ftNum+1)

        # print self.ftNum
        # print self.dataNum
        # J
        J = self.costFunc()
        
        print "\n J cost function data---------------------------------------"
        print J
        print self.thetas

        while not converged:
            # print self.thetas
            idx = 0
            # Get thetas
            for j in range(0, self.ftNum):
                # Do we need to devide the number of datas
                self.thetas[j] = self.thetas[j] - alpha * \
                    self.wgtVals(j) * (1/self.dataNum)

            e = self.costFunc()
            print "\n E cost function data---------------------------------------"
            print e
            print self.thetas

            if abs(J-e) <= ep:
                print 'Converged, iterations: ', count, '!!!'
                converged = True
            J = e
            count += 1
            if count == max_iter:
                print 'Max interactions exceeded!'
                converged = True

        return self.thetas


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
            # print arr
            if True:
                arr[2] = float(datetime.strptime(
                    arr[2], '%m/%d/%Y').strftime("%Y%m%d"))

            y.append(float(arr.pop().replace("\n", "")))
            # x.append(map(float, arr))
            # x.append([arr[3]])
            x.append([float(arr[3]),float((arr[3]))**2])
            return [x, y]       

            

    return [x, y]


"""open csv"""
# devData = pd.read_csv('PA1_dev.csv', header = None)
# tstData = pd.read_csv('PA1_test.csv', header = None)
# trnData = pd.read_csv('PA1_train.csv', header = None)

print "\n ---------------------------------------"
dataSet = importCsv("PA1_train (1).csv", ",")
print np.random.random()

print "\n Import Data testing datas, result datas---------------------------------------"
print dataSet[0]
print ""
print dataSet[1]

print "\n LinearRegression ---------------------------------------"
lg = LinearRegression(dataSet[0], dataSet[1])
w = lg.gradientDescent()
print w


# reader = csv.reader(open("PA1_test.csv", "rb"), delimiter=",")
# result = np.array(list(reader))
# print len(result[0])
# print result[1:]

# theta = gradient_descent(alpha, x, y, ep, max_iter=1000)
