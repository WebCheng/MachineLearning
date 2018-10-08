# date to YYYYMMDD
#

import numpy as np
import matplotlib.pyplot as plt
import helper as hp


# Linear Regression class
class LinearRegression:
    def __init__(self, datas, result):
        self.x = datas              # training examples
        self.y = result             # outcome price
        self.dataNum = len(datas)   # number of data
        self.ftNum = len(self.x[0])  # number of Xi(feature)
        self.thetas = []            # array of parameters

    # Initilize parameters
    def initTheta(self, nums):
        rt = []
        for i in range(0, nums):
            rt.append(1)
        return rt

    # Cost Function
    def costFunc(self):
        J = 0
        for i in range(0, self.dataNum):
            tmp = 0.0
            for j in range(0, self.ftNum):
                tmp = tmp + self.thetas[j] * self.x[i][j]

            J += ((1.0/(2.0*self.dataNum))*(tmp - self.y[i])**2)
        return J

    # idx : for specific X value
    def wgtVals(self, idx):
        rt = 0
        # Sum up all the values
        for i in range(0, self.dataNum):
            # compute (X^T*theta)
            tmp = 0
            for j in range(0, self.ftNum):
                tmp = tmp + self.thetas[j] * (self.x[i][j])
            rt += (tmp - self.y[i]) * self.x[i][idx]
        return rt

    # Regulization
    def regVal(self, lam, theta):
        return (lam/self.dataNum) * theta

    # Gradient Descent
    def gradientDescent(self, alpha=10**-1, limit=0.001, maxIter=1000, lam=0.0):

        converged, count = False, 0
        self.thetas = self.initTheta(self.ftNum)

        J = self.costFunc()

        while not converged:

            idx, tmp = 0, []
            for j in range(0, self.ftNum):
                gdVal = self.wgtVals(j) * (1.0/self.dataNum)
                tmp.append(self.thetas[j]
                           - (alpha * gdVal + self.regVal(lam, self.thetas[j])))

            self.thetas = tmp

            e = self.costFunc()
            count += 1
            if abs(J-e) <= limit:  # ???
                print('Converged, iterations: ', count, '!!!')
                converged = True

            J = e
            if count == maxIter:
                print('Max interactions exceeded!')
                converged = True

        return self.thetas


# =============================================================================
# ################ Main Function ################
# =============================================================================
alphaVal = 10 ** (-1)   # learning rate
limit = 0.001           # convergence condition
ep = 0.5                # norm of the gradient
maxIter = 10000         # limitation of iteration
lam = 0.0               # regularization coefficient


"""open csv"""
print("\n ------------ ImportDaTa ------------")
dataSet = hp.importCsv("./Document/PA1_train.csv")
testSet = hp.importCsv("./Document/PA1_dev.csv")


print("\n ------------ LinearRegression ------------")
lg = LinearRegression(dataSet[0], dataSet[1])
w = lg.gradientDescent(alphaVal, ep, maxIter, lam)
print(w)

print "\n --------------------------------------- SEE Compute ---------------------------------------"
print hp.sse(testSet[1], hp.predictVals(w, testSet[0]))

""" drawing plot """
# arr = [x[1] for x in dataSet[0]]
# plt.scatter(arr, dataSet[1])
# plt.ylabel('y lable')
# plt.xlabel('x lable')
# plt.show()
