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


def predictVals(thetas, datas):
    rt = np.array([])
    for row in datas:
        predVal = np.dot(row, thetas)
        rt = np.append(rt, predVal)
    return rt


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
            
            
            arr[2] = float(datetime.strptime( arr[2], '%m/%d/%Y').strftime("%Y%m%d"))
            year = float(arr[2]/10000)
            month = float(arr[2]/100%100)
            day = float(arr[2]%100)

            y.append(float(arr.pop().replace("\n", "")))
            
            x.append([float(arr[0])
            ,year/1000
            ,month/10
            ,day/10
            # ,float(arr[2])
            # ,float(arr[3])
            # ,float(arr[4])
            # ,float(arr[5])/1000
            # ,float(arr[6])/1000
            # ,float(arr[7])
            # ,float(arr[8])
            # ,float(arr[9])
            # ,float(arr[10])
            # ,float(arr[11])
            # ,float(arr[12])/1000
            # ,float(arr[13])/1000
            # ,float(arr[14])/1000
            # ,float(arr[15])/1000
            # ,float(arr[16])/10000
            # ,float(arr[17])/10
            # ,float(arr[18])/(-100)
            # ,float(arr[19])/1000
            ])

    return [x, y]
