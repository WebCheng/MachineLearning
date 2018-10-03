import csv
import pandas as pd
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score




# initial ramdon theta
def initialTheta(nums):
    thetas = []
    for i in range(0,nums):
        thetas.append(i)

    return thetas

def computeCostFunc(x, y ,theta,dataNum,ftNum):
    J = 0
    for i in range(0,dataNum):
        tmp = 0
        for j in range(0,ftNum):
            tmp += theta[j]*x[j]
        J+=(tmp - y[i])**2
    return J/2



def gradient_descent(alpha, x, y, ep=0.0001, max_iter=10000):
    converged = False
    iter = 0

    featureNumber = 21
    # number of samples 
    dataNum = len(y)   
    # initial theta(feature numbers)
    theta = initialTheta(featureNumber)
    # J(theta) :cost function
    J=computeCostFunc(x,y,theta,dataNum,featureNumber) 

    # Iterate Loop
    while not converged:
        # for each training sample, compute the gradient (d/d_theta j(theta))
        grad0 = sum([(t0 + t1*x[i] - y[i]) for i in range(m)]) 
        grad1 = sum([(t0 + t1*x[i] - y[i])*x[i] for i in range(m)])

        # update the theta_temp
        temp0 = t0 - alpha * grad0
        temp1 = t1 - alpha * grad1
    
        # update theta
        t0 = temp0
        t1 = temp1

        # mean squared error
        e = sum( [ (t0 + t1*x[i] - y[i])**2 for i in range(m)] ) 

        if abs(J-e) <= ep:
            print 'Converged, iterations: ', iter, '!!!'
            converged = True
    
        J = e   # update error 
        iter += 1  # update iter
    
        if iter == max_iter:
            print 'Max interactions exceeded!'
            converged = True

    return ans

def importCsv(path,delimiter,isHead = True ):
    x,y = [],[]
    ftrNum = 0
    
    with open(path) as f:
        lines = f.readlines()

        for line in lines:

            if isHead:   
                isHead = False
                continue
 
            arr = line.split(delimiter)
            y.append(arr.pop().replace("\n",""))
            x.append(arr)
            ftrNum = len(arr)
            
    print x
    print y
    print ftrNum

"""open csv""" 
# devData = pd.read_csv('PA1_dev.csv', header = None)  
# tstData = pd.read_csv('PA1_test.csv', header = None)    
# trnData = pd.read_csv('PA1_train.csv', header = None)   
 
importCsv("PA1_test.csv",",")

# reader = csv.reader(open("PA1_test.csv", "rb"), delimiter=",")
# result = np.array(list(reader)) 
# print len(result[0])
# print result[1:] 

# theta = gradient_descent(alpha, x, y, ep, max_iter=1000)