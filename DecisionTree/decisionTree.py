import math
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def count(nums):
    rt, sumNum = 0, np.sum([nums[i] for i in range(len(nums))])
    for i in range(len(nums)):
        if nums[i] != 0:
            p = nums[i] / sumNum
            rt += (-p * math.log(p, 2))
    return rt


def addDicValue(dic, key):
    dic.setdefault(key, 0)
    dic[key] += 1


def count_Info(arr):
    dic = {}
    for i in range(arr.shape[1]):
        addDicValue(dic, arr[0, i])
    return [dic[x] for x in dic.keys()]


def count_pp(dic, key, data):
    dic.setdefault(key)
    tmp = {} if dic[key] is None else dic[key]
    for i in range(data.shape[1]):
        addDicValue(tmp, data[0, i])
    dic[key] = tmp


def buildTree(mat):
    dataNum = mat.shape[0]

    # SELECT Y, COUNT(Y) FROM TABLE GROUP BY Y
    nums = count_Info(mat.T[-1])
    if len(nums) == 1:
        return {}, None
    vals, dic, maxIdx, maxInfoGainNum, hy = [], {}, 0, 0, count(nums)

    # print(mat.T)
    # for attributeArr in np.hsplit(mat.T, [nums[0]]):
    #     for i in range(0, attributeArr.shape[0] - 1):
    #         count_pp(dic, i, attributeArr[i,])
    #         vals.append(count(count_Info(attributeArr[i,])))

    for col in range(mat.shape[1] - 1):
        valDic = {}
        for row in range(mat.shape[0]):
            key = col
            dic.setdefault(key)
            tmp = {} if dic[key] is None else dic[key]
            tmp.setdefault(mat[row, col], 0)
            tmp[mat[row, col]] += 1
            dic[key] = tmp

            valDic.setdefault(mat[row, col])
            tmp = {} if valDic[mat[row, col]] is None else valDic[mat[row, col]]
            k = mat[row, mat.shape[1] - 1]
            tmp.setdefault(k, 0)
            tmp[k] += 1
            valDic[mat[row, col]] = tmp

        for k, v in valDic.items():
            dd = []
            for att, num in v.items():
                dd.append(num)
            vals.append(count(dd))

    # # For the P values (How many data will distribute in this situation?)
    # # Each attributions will have their own P value.
    # print(dic)
    # # For the H(Y|Xi = True) and H(Y|Xi = False)
    # print(vals)
    for i in range(0, mat.shape[1] - 1):

        # tt = 0
        # for k in dic[0].keys():
        #     tt += (dic[i][k] / dataNum) * vals[i]
        # tmp = hy - tt
        # print("{0} = {1} - {2}".format(tmp, hy, tt))

        tmp = hy - ((dic[i][0] / dataNum) * vals[2 * i] + (dic[i][1] / dataNum) * vals[(2 * i + 1)])
        print("{0} = {1} - {2} - {3}".format(tmp, hy, (dic[i][0] / dataNum) * vals[2 * i],
                                             (dic[i][1] / dataNum) * vals[(2 * i + 1)]))
        if tmp > maxInfoGainNum:
            maxIdx, maxInfoGainNum = i, tmp

    # Split the data depend on the FEATURE NUMBERS
    splitDic = {}
    for key in dic[maxIdx].keys():
        splitDic.setdefault(key)
        # Initail None Empty Arr
        splitDic[key] = np.matrix([[0 for i in range(0, mat.shape[1] - 1)]])

        # Compare Every Data Number For Spliting
        for i in range(dataNum):
            if key == mat[i, maxIdx]:
                splitDic[key] = np.insert(splitDic[key], splitDic[key].shape[0], np.delete(mat[i], [maxIdx]), axis=0)
        # Specific to remove first [0000] arr
        splitDic[key] = splitDic[key][1:]
    return splitDic, maxIdx


def dicitionTree(mat):
    splitData, maxIdx = buildTree(mat)
    for v in splitData.values():
        dicitionTree(v)


values = [[0, 1, 1, 0], [1, 1, 1, 0], [0, 0, 0, 0], [1, 1, 0, 1], [0, 1, 0, 1], [1, 0, 1, 1]]
matrix = np.matrix(values)
dicitionTree(matrix)
