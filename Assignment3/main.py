import helper as hp
import decisionTree as dt
import random_forest as ft
import adaboost as ada
from binarytree import Node
import datetime
import warnings

# =============================================================================
# ################ Main Function ################
# =============================================================================
maxDepth = 20  # Online\Avg Perceptron Loop number
fileName1 = "pa3_train_reduced_bo.csv"  # Training File name
fileName2 = "pa3_valid_reduced.csv"  # Validate File name
warnings.filterwarnings("error")
print("\n ------------ ImportDaTa ------------")
trainData = hp.importCsv(fileName1)
validateData = hp.importCsv(fileName2)

for l in [1, 5, 10, 20]:
# for l in [1]:
    print("\n ------------ Adaboost-{0} ------------{1}".format(l, datetime.datetime.now()))
    adaClass = ada.adaboost(ftrNum=trainData.shape[1] - 1, depth=1, lNum=l, dataNum=trainData.shape[0])
    adaClass.runAdaboost(df=trainData)
    print(adaClass.computeFinalAccNumRate(df=trainData))

# for d, m, n in [(9, 20, 1), (9, 20, 2), (9, 20, 5), (9, 20, 10), (9, 20, 25)]:
# for d, m, n in [(9, 50, 1), (9, 50, 2), (9, 50, 5), (9, 50, 10), (9, 50, 25)]:
# for d, m, n in [(9, 10, 1), (9, 10, 2), (9, 10, 5), (9, 10, 10), (9, 10, 25)]:
#     print("\n ------------ Build Forest{0} ------------{1}".format(n, datetime.datetime.now()))
#     ftClass = ft.randomForest(treeNum=n, ftrNum=m, depth=d, dataNum=trainData.shape[0])
#     ftClass.buildRandomForest(df=trainData)
#     ftClass.predicDataResult(df=trainData)

# print("\n ------------ Build DT ------------{0}".format(datetime.datetime.now()))
# dtClass = dt.decesionTree(maxDepth, trainData.shape[0])
# root1 = root2 = cur = Node((None, None, dtClass.getLabelFromLargeData(trainData)))
# cl, cr = dtClass.getResultInfo(trainData)
# dtClass.dicisionTree(trainData, cur, 0, cl, cr)
#
# print("\n ------------ Acc - TrainingData ------------{0}".format(datetime.datetime.now()))
# dtClass.predictRec(trainData, root1)
# trnAccNum = dtClass.getAccNum()
# print(trnAccNum / trainData.shape[0])
# dtClass.resetAccNum()
#
# print("\n ------------ Acc - ValidateDaTa ------------{0}".format(datetime.datetime.now()))
# dtClass.predictRec(validateData, root2)
# validateAccNum = dtClass.getAccNum()
# print(validateAccNum / validateData.shape[0])
# dtClass.resetAccNum()
