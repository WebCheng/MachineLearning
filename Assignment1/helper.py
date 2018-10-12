from datetime import datetime
import numpy as np 
import math


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

"""
Normalize Values between 0 to 1
Matrix -> Matrix
"""


def NormalizeData(matrix):
    mins = np.min(matrix, axis=0)
    maxs = np.max(matrix, axis=0)
    maxs - mins

    for col in range(1, len(matrix[0])):
        for row in range(0, len(matrix)):
            matrix[row][col] = (matrix[row][col] - mins[col]
                                ) / (maxs[col] - mins[col])

    return matrix


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

            year = float(math.floor(arr[2]/10000))
            month = float(math.floor((arr[2]-year*10000) / 100))
            day = float(arr[2]-year*10000-month*100)
#            print("Y/M/D: ", year, month, day) # for testing

            diff_day = (2018*365 + 5*30 + 31) - (year*365 + month*30 + day)

            y.append(float(arr.pop().replace("\n", "")))
            x.append([float(arr[0])  
                    #   ,float(arr[1])
                      #   ,float(arr[2])
                        ,diff_day
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
                        ,float(arr[20])
                      ])
    
    return [NormalizeData(x), y]
