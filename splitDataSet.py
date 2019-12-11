#将每个数据集提取20%，各生成10份
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

filename_list = [ 'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv', 'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv', 'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv', 'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv', 'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv', 'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv']
x = 0
for x in range(0,10):

    print("---------------------this is file "+str(x+1)+"---------------------------")
    dataSet = pd.read_csv(filename_list[x])

    labelData = dataSet['Label']
    labelData = np.array(labelData)


    ss=StratifiedShuffleSplit(n_splits=10,test_size=0.2,train_size=0.8,random_state=0)#分成10组，测试比例为0.2，训练比例是0.8
    i=0
    for train_index, test_index in ss.split(labelData, labelData):
        print("TRAIN:", train_index, "TEST:", test_index)#获得索引值
        i=i+1
        print(i)
        #index_arr = np.array(test_index)
        #index_csv = pd.DataFrame(index_arr.T)
        #index_csv.to_csv("index.csv",header=False,index=False,mode='a')
        #X_train, X_test = X[train_index], X[test_index]#训练集对应的值
        #y_train, y_test = y[train_index], y[test_index]#类别集对应的值
        dt = dataSet.loc[test_index]
        filename = "split" + str(i) + "_" + filename_list[x]
        #filename = "split10_" + filename_list[x]
        dt.to_csv(filename,index=False)
    x = x+1
