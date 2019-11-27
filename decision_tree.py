import time
import pandas as pd
import numpy as np
from my_logger import set_logger
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import classification_report

logger = set_logger('dt')
for i in range(1, 11):
    switch = 0
    logger.info(f'Reading data {i}...')
    if switch == 1:
        data = pd.read_csv(f"E:\\data\\result\\new_dataset_{i}.csv")
    elif switch == 0:
        data = pd.read_csv(f"E:\\data\\result\\dataset_random_split{i}_modified.csv")
    logger.info('Done!')
    encoder = LabelEncoder()
    data['Label'] = encoder.fit_transform(data['Label'])

    logger.info('Splitting data set...')
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=int(time.time()))
    for train_index, test_index in split.split(data, data['Label']):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]
    logger.info('Done!')

    logger.debug(data['Label'].value_counts())
    logger.debug(strat_train_set['Label'].value_counts())
    if switch == 0:
        dtc = DecisionTreeClassifier(criterion='entropy', max_features=None)
        # dtc = RandomForestClassifier(n_estimators=20, criterion='entropy', max_features=10, bootstrap=False)
    elif switch == 1:
        dtc = DecisionTreeClassifier(criterion='entropy', max_features=None)
    X_train = strat_train_set[data.columns[:-1]].to_numpy()
    y_train = strat_train_set['Label'].to_numpy()
    X_test = strat_test_set[data.columns[:-1]].to_numpy()
    y_test = strat_test_set['Label'].to_numpy()
    logger.info('Training...')
    start = time.time()
    dtc.fit(X_train, y_train)
    end1 = time.time()
    logger.info(f'Training complete. Time elapsed: {(end1 - start)}s')
    y_predict = dtc.predict(X_test)
    end2 = time.time()
    logger.info(f'Test complete. Time elapsed: {(end2 - start)}s')
    logger.info('Confusion Matrix:')
    print(confusion_matrix(y_test, y_predict, labels=pd.unique(y_test)))
    logger.info(f'Accuracy:  {accuracy_score(y_test, y_predict)}')
    logger.info(f'Precision: {precision_score(y_test, y_predict, average="macro")}')
    logger.info(f'Recall : {recall_score(y_test, y_predict, average="macro")}')
    logger.info(f"F1 score: {f1_score(y_test, y_predict, average='macro')} ")
    logger.info('Feature Importances:')
    print(dtc.feature_importances_)
    print(classification_report(y_test, y_predict, target_names=encoder.classes_))
