# date to YYYYMMDD
#

import numpy as np
import matplotlib.pyplot as plt
import helper as hp


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
            rt.append(1)
            # rt.append(np.random.random())
        return rt

    # Cost Function
    def costFunc(self):
        J = 0
        for i in range(0, self.dataNum):
            tmp = 0.0
            for j in range(0, self.ftNum):
                tmp = tmp + self.thetas[j] * self.x[i][j]
            try:
                J += ((1.0/(2.0*self.dataNum))*(tmp - self.y[i])**2)
            except:
                print tmp
                print self.y[i]

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

    def gradientDescentReg(self, alpha=10**-1, ep=0.001, max_iter=1000, lam=0):

        count = 0
        converged = False
        self.thetas = self.initTheta(self.ftNum+1)

        # J
        J = self.costFunc()

        while not converged:
            
            idx, tmp = 0, []
            for j in range(0, self.ftNum):
                tmp.append(self.thetas[j] - (alpha *
                                             self.wgtVals(j) * (1.0/self.dataNum)))
            self.thetas = tmp

            e = self.costFunc()

            if abs(J-e) <= ep:
                print 'Converged, iterations: ', count, '!!!'
                converged = True

            J = e
            count += 1
            if count == max_iter:
                print 'Max interactions exceeded!'
                converged = True

        return self.thetas

    def gradientDescent(self, alpha=10**-1, ep=0.001, max_iter=1000):

        count = 0
        converged = False
        self.thetas = self.initTheta(self.ftNum+1)

        # J
        J = self.costFunc()

        while not converged:
             
            idx, tmp = 0, []
            for j in range(0, self.ftNum):
                tmp.append(self.thetas[j] - (alpha *
                                             self.wgtVals(j) * (1.0/self.dataNum)))
            self.thetas = tmp

            e = self.costFunc()

            if abs(J-e) <= ep:
                print 'Converged, iterations: ', count, '!!!'
                converged = True

            J = e
            count += 1
            if count == max_iter:
                print 'Max interactions exceeded!'
                converged = True

        return self.thetas


################ Main Function ################
alphaVal = 10 ** (-3)


"""open csv"""
print "\n --------------------------------------- ImportDaTa ---------------------------------------"
dataSet = hp.importCsv("./Document/PA1_train.csv", ",")

print "\n --------------------------------------- LinearRegression ---------------------------------------"
lg = LinearRegression(dataSet[0], dataSet[1])
w = lg.gradientDescent()
print w

""" drawing plot """
# arr = [x[1] for x in dataSet[0]]
# plt.scatter(arr, dataSet[1])
# plt.ylabel('y lable')
# plt.xlabel('x lable')
# plt.show()
