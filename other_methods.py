import time
import pandas as pd
import numpy as np
from TimeFormatter import time_formatter
from my_logger import set_logger
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif, RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

logger = set_logger('chi2', 'other_methods.log')
# logger.info(f'Using Method: {method.__name__}')
for i in range(5, 6):
    logger.info(f'Reading data {i}...')
    data = pd.read_csv(f"E:\\data\\result\\dataset_random_split{i}_modified.csv")
    logger.info('Done!')
    columns = data.columns[:-1]
    X = data[columns]
    y = data['Label']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    selector = SelectKBest(chi2, k=10)
    try:
        start = time.time()
        logger.info('Selecting features...')
        selector.fit(X, y)
        end = time.time()
        logger.info(f'Done! Time elapsed: {time_formatter(end - start)}')
        support = selector.get_support(indices=True)
        logger.info(support)
        X_new = selector.transform(X)
        # print(X_new.shape)
        # print(selector.scores_)
        # print(X_new)
        logger.info([data.columns[j] for j in support])

        clf = DecisionTreeClassifier(criterion='entropy')
        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(time.time()))
        for train_index, test_index in split.split(X_new, y):
            print('TRAIN: ', train_index, 'TEST: ', test_index)
            X_train, X_test = X_new[train_index], X_new[test_index]
            y_train, y_test = y[train_index], y[test_index]

        # print(X_train)
        # print(y_train)
        # print(X_test)
        # print(y_test)
        logger.info('Training...')
        start = time.time()
        clf.fit(X_train, y_train)
        end1 = time.time()
        logger.info(f'Training complete. Time elapsed: {(end1 - start)}s')
        y_predict = clf.predict(X_test)
        end2 = time.time()
        logger.info(f'Test complete. Time elapsed: {(end2 - start)}s')
        logger.info('Confusion Matrix:')
        logger.info(f'\n{confusion_matrix(y_test, y_predict, labels=pd.unique(y_test))}')
        logger.info(f'Accuracy:  {accuracy_score(y_test, y_predict)}')
        logger.info(f'Precision: {precision_score(y_test, y_predict, average="macro")}')
        logger.info(f'Recall : {recall_score(y_test, y_predict, average="macro")}')
        logger.info(f"F1 score: {f1_score(y_test, y_predict, average='macro')} ")
    except ValueError as e:
        logger.warning(f'File {i}: ' + str(e))
