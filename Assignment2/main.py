import numpy as np
import helper as hp
import datetime as dt
import perceptron as pct
import kernelPerceptron as kpt
 
maxIter = 15
fileName1 = "pa2_train.csv"      # File name one
fileName2 = "pa2_valid.csv"      # File name one

print("\n ------------ ImportDaTa ------------")
(par1, rst1) = hp.importCsv(fileName1)
(par2, rst2) = hp.importCsv(fileName2)
hp.setLabels(rst1, 3, 5)
hp.setLabels(rst2, 3, 5)
par1 = np.matrix(par1)
rst1 = np.matrix(rst1)
par2 = np.matrix(par2)
rst2 = np.matrix(rst2)
print(par1.shape)
print(par2.shape)

print("\n ------------ Perceptron ------------")
pt = pct.Perceptron(par1, rst1, par2, rst2)
w = pt.onlinePerceptron(15)

print("\n ------------ AvgPerceptron ------------")
pt = pct.Perceptron(par1, rst1, par2, rst2)
w = pt.avgPerceptron(15)


print("\n ------------ kerPerceptron ------------")
print(dt.datetime.now())
kp = kpt.KernelPerceptron(par1, rst1, par2, rst2)
w = kp.kernelPerceptron(10)
print(dt.datetime.now())
