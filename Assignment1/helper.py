from datetime import datetime
import numpy as np
import pandas as pd

"""
Get SSE given ground truth and prediction vectors
"""


def sse(realVal, predVal):
    return sum((realVal-predVal)**2)/2.0


"""
Takes in weights vector w, and a dataset matrix which contains input rows excluding output column
and returns prediction vector.
"""


def predictVals(thetas, datas):
    rt = np.array([])
    for row in datas:
        predVal = np.dot(row, thetas)
        rt = np.append(rt, predVal)
    return rt


def NormalizeDatas(matrix):
    df = pd.DataFrame(matrix)
    dfNorm = (df - df.min()) / (df.max() - df.min())
    
    print("\n ------------ ImportDaTa static detail value------------")
    print dfNorm.describe()

    return dfNorm.values



"""
Import data from CSV files
"""


def importCsv(path, delimiter=",", isHead=True):
    x, y = [], []

    with open(path) as f:
        lines = f.readlines()

        for line in lines:

            if isHead:
                isHead = False
                continue

            arr = line.split(delimiter)

            arr[2] = float(datetime.strptime(
                arr[2], '%m/%d/%Y').strftime("%Y%m%d"))
            year = float(arr[2]/10000)
            month = float(arr[2]/100 % 100)
            day = float(arr[2] % 100)

            y.append(float(arr.pop().replace("\n", "")))

            x.append([float(arr[1])
                      ,float(arr[2])
                      ,float(arr[3])
                      ,float(arr[4])
                      ,float(arr[5])
                      ,float(arr[6])
                      ,float(arr[7])
                      ,float(arr[8])
                      ,float(arr[9])
                      ,float(arr[10])
                      ,float(arr[11])
                      ,float(arr[12])
                      ,float(arr[13])
                      ,float(arr[14])
                      ,float(arr[15])
                      ,float(arr[16])
                      ,float(arr[17])
                      ,float(arr[18])
                      ,float(arr[19])
                      ])
    
    return [NormalizeDatas(x), y]
    # return [x, y]
