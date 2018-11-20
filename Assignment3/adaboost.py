import decisionTree as dt
import numpy as np
from binarytree import Node
import math
import datetime
import copy


class adaboost:

    def __init__(self, ftrNum, depth, lNum, dataNum):
        self.wArr = []
        self.ftrNum = ftrNum
        self.maxDepth = depth
        self.dataNum = dataNum
        self.L = lNum
        self.dtClass = dt.decesionTree(depth=self.maxDepth, dataNum=self.dataNum)
        self.alphaSet = []
        self.treeSet = []
        self.weightSet = []

    def getAccNum(self, df, root):
        self.dtClass.resetAccNum()
        self.dtClass.predictRec(df, root)
        return self.dtClass.getAccNum()

    def getAlphaVal(self, errRate):
        return (1 / 2) * math.log((1 - errRate) / errRate)

    def runAdaboost(self, df):
        self.wArr = np.ones(df.shape[0])
        cl, cr = self.dtClass.getResultInfo(df)
        val = self.dtClass.getLabelFromLargeData(df)

        for l in range(0, self.L):
            self.weightSet.append(copy.deepcopy(self.wArr))

            root = cur = Node((None, None, val))
            self.learn(df, cl, cr, cur)

            alpha = self.computeAlpha(df, root, l)
            self.modifiedWeight(df, root, alpha)
            self.alphaSet.append(alpha)
            self.treeSet.append(root)

    def learn(self, df, cl, cr, currentRoot):
        self.dtClass.setDataWeight(self.wArr)
        self.dtClass.dicisionTree(df, currentRoot, 0, cl, cr)

    def computeAlpha(self, df, root, l):
        totalNum = sum(self.wArr)
        errRate = (totalNum - self.getAccNum(df, root)) / totalNum
        print("Error Rate = {0}".format(errRate))
        if errRate == 0:
            print("Loop {0} times find error rate = 0".format(l + 1))
        return self.getAlphaVal(errRate)

    def modifiedWeight(self, df, root, alpha):
        self.dtClass.resetAccNum()
        for dataIdx in range(0, df.shape[0]):
            self.dtClass.predictRec(df.iloc[dataIdx:dataIdx + 1, :], root)
            if self.dtClass.getAccNum() != 0:
                self.wArr[dataIdx] *= math.exp(-alpha)
            else:
                self.wArr[dataIdx] *= math.exp(alpha)
            self.dtClass.resetAccNum()
        return alpha

    def computeFinalAccNumRate(self, df):
        fAccNum, dataNum = 0, df.shape[0]
        for dataIdx in range(0, dataNum):
            weight, tmp = 0, 1
            for i in range(0, self.L):
                self.dtClass.predictRec(df.iloc[dataIdx:dataIdx + 1, :], self.treeSet[i])
                isCorrect = 1 if self.dtClass.getAccNum() != 0 else -1

                # maybe we dont need to multiply weightSet[i][dataIdx]
                # It should be computed by using alpha value
                tmp += isCorrect * self.alphaSet[i] * self.weightSet[i][dataIdx]
                self.dtClass.resetAccNum()
            if np.sign(tmp) > 0:
                fAccNum += 1
        return fAccNum / dataNum
