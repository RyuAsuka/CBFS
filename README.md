# CBFS: A Clustering-Based Feature Selection Mechanism for Network Anomaly Detection
The source code of my paper _Mao, J., Hu, Y., Jiang, D., Wei, T., & Shen, F. (2020). CBFS: A Clustering-Based Feature Selection Mechanism for Network Anomaly Detection. IEEE Access, 8, 116216-116225._.

## Python programme files

* splitDataSet.py: Extract 20% of CIC-IDS-2018 data set, and repeat 10 times.
* combine.py: Combine all extracted data set to one file.
* FeatureClustering.py: The feature clustering algorithms in my paper.
* InformationGainRanking.py: The ranking algorithm in my paper.
* my_logger.py: An assistance class which encapsulate the `logging` class in Python.
* TimeFormatter.py: An assistance class to unify the time format.

## Usage

Data provider: [https://www.unb.ca/cic/datasets/ids-2018.html](https://www.unb.ca/cic/datasets/ids-2018.html)

The preprocessing functions are not provided. Please do it yourself by `scikit-learn` library.

Then run `FeatureClustering.py` to cluster all features you extract and generate a batch of file with different distance threshold.

Next, run `InformationGainRanking.py` to use information gain and gain ratio to rank the clustered features comprehensivly.

The training and testing code are not provided. Please use `sklearn.tree.DecisionTreeClassifier` class to train the decision tree and test at your own.