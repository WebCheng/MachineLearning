from datetime import datetime
import numpy as np


"""
Get SSE given ground truth and prediction vectors
"""


def sse(realVal, predVal):
    return sum((realVal-predVal)**2)


"""
Takes in weights vector w, and a dataset matrix which contains input rows excluding output column
and returns prediction vector.
"""


def makePrediction(thetas, datas):
    rt = np.array([])
    for row in datas:
        predVal = np.dot(row, thetas)
        rt = np.append(rt, predVal)
    return rt


"""
Import data from CSV files
"""


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
            x.append([float(arr[0]), float(arr[2])*(10**-9)])

    return [x, y]
