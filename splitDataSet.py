"""
Extract 20% of every data set file, and generate 10 data sets.
"""
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd

filename_list = [
    'Friday-02-03-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-16-02-2018_TrafficForML_CICFlowMeter.csv',
    'Friday-23-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thuesday-20-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-15-02-2018_TrafficForML_CICFlowMeter.csv',
    'Thursday-22-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-21-02-2018_TrafficForML_CICFlowMeter.csv',
    'Wednesday-28-02-2018_TrafficForML_CICFlowMeter.csv']

for x in range(0, 10):
    print("--------------------- File: " + filename_list[x] + "---------------------------")
    dataSet = pd.read_csv(filename_list[x])

    label_data = dataSet['Label']
    label_data = np.array(label_data)

    ss = StratifiedShuffleSplit(
        n_splits=10,
        test_size=0.2,
        train_size=0.8,
        random_state=0
    )  # Split to 10 groups, test_size = 0.2, train_size = 0.8
    i = 0
    for train_index, test_index in ss.split(label_data, label_data):
        print("TRAIN:", train_index, "TEST:", test_index)  # Get the index values
        i = i + 1
        print(i)
        dt = dataSet.loc[test_index]
        filename = "split" + str(i) + "_" + filename_list[x]
        dt.to_csv(filename, index=False)
