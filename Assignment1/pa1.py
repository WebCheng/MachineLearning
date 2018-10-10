# date to YYYYMMDD
#
import numpy as np 
import helper as hp
from datetime import date

# Linear Regression class
class LinearRegression:
    def __init__(self, datas, result):
        self.x = datas              # training examples
        self.y = result             # outcome price
        self.dataNum = len(datas)   # number of data
        self.ftNum = len(self.x[0]) # number of Xi(feature)
        self.thetas = []            # array of parameters

    # Initilize parameters
    def initTheta(self, nums):
        rt = []
        for i in range(0, nums):
            rt.append(0)
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
    def gradientDescent(self, alpha=10**-1, limit=0.5, maxIter=1000, lam=0.0):

        converged, count = False, 0
        self.thetas = self.initTheta(self.ftNum)

        while not converged:

            isAllCon, idx, tmp = True, 0, []
            for j in range(0, self.ftNum):
                gdVal = self.wgtVals(j) * (1.0/self.dataNum)

                tmp.append(self.thetas[j]
                           - (alpha * gdVal + self.regVal(lam, self.thetas[j])))

                isAllCon = False if abs(gdVal) > limit else isAllCon

            count += 1
            converged = True if count == maxIter else isAllCon
            self.thetas = tmp if converged == False else self.thetas

        print('Converged, iterations: ', count, '!!!')

        return self.thetas


# =============================================================================
# ################ Main Function ################
# =============================================================================
alphaVal = 10 ** (-7)   # learning rate
limit = 0.5             # convergence condition
maxIter = 1000          # limitation of iteration
lam = 0.0               # regularization coefficient
  
"""open csv"""
print("\n ------------ ImportDaTa ------------")
dataSet = hp.importCsv("./Document/PA1_train.csv")
# testSet = hp.importCsv("./Document/PA1_dev.csv")


print("\n ------------ LinearRegression ------------")
lg = LinearRegression(dataSet[0], dataSet[1])
print date.today()
w = lg.gradientDescent(alphaVal, limit, maxIter, lam)
print(w)

print("\n --------------------------------------- SSE Compute ---------------------------------------")
print(hp.sse(dataSet[1], hp.predictVals(w, dataSet[0])))

# """ drawing plot """
# arr = [x[1] for x in dataSet[0]]
# plt.scatter(arr, dataSet[1])
# plt.ylabel('y lable')
# plt.xlabel('x lable')
# plt.show()
