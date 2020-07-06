import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from my_logger import set_logger

logger = set_logger('ranking', 'ranking2.log')

NEED_DISCRETIZE = False


def entropy(df, column, nrows):
    """
    Calculate the entropy of a certain column.

    Parameters
    ----------
        df : pandas.DataFrame
            The data frame of input data.
        column : str
            One column of data.
        nrows : int
            The shape (rows) of the data.

    Returns
    -------
        float :
            The entropy value.
    """
    value_counts = df[column].value_counts().astype('float64')
    value_prob = value_counts / nrows
    logged_value_prob = value_prob * np.log2(value_prob)
    return -logged_value_prob.sum()


def get_information_gain(df, label_name, nrows):
    """
    Calculate the information gain of each feature.

    Parameter
    ---------
        df : pandas.DataFrame
            The data set organized by pandas.DataFrame.
        label_name : str
            The name of label column
        nrows : int
            The number of rows of the data set.

    Returns
    -------
        dict[str:float] :
            The information gain of each feature. The key is feature name and the value is its information gain.
    """
    info_gains = {}
    ent_label = entropy(df, label_name, nrows)
    for column in tqdm(df.columns[:-1], total=len(df.columns) - 1, ncols=80):
        # for column in df.columns[:-1]:
        groups = df.groupby(column)
        column_prob = df[column].value_counts().astype('float64') / nrows
        tablen = column_prob.shape[0]
        cond_ent = 0.0
        for name, group in tqdm(groups, total=tablen, ncols=80, desc=column):
            # for name, group in groups:
            group_nrows = group.shape[0]
            prob_label = group[label_name].value_counts().astype('float64') / group_nrows
            logged_prob_label = prob_label * np.log2(prob_label)
            try:
                ent = column_prob[name] * logged_prob_label.sum()
            except KeyError:
                ent = 0.0
            cond_ent -= ent
        info_gains[column] = ent_label - cond_ent
    return info_gains


def discretizer(dataframe):
    """
    Discretize the data set.

    Parameters
    ----------
        dataframe : pandas.DataFrame
            The original data frame.

    Returns
    -------
        pandas.DataFrame:
            The converted data frame.
    """
    for col in dataframe.columns[:-1]:
        if col == 'Dst Port':
            bins = [0, 1000, 10000, 32768, 65536]
            dataframe[col] = pd.cut(dataframe[col], bins=bins, right=False, labels=False)
        else:
            dataframe[col] = pd.cut(dataframe[col], bins=100, labels=False)
    return dataframe


def check_data(data, label):
    return data.shape[0] == label.shape[0]


if __name__ == '__main__':
    label_type = 'binary'  # 'binary' or 'multiclass'
    for delta in [0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        logger.info(f'Reading data set with delta={delta}...')
        data_set_file = f"X_train_{delta}.csv"
        if label_type == 'binary':
            label_file = f"y_train_binary.csv"
        elif label_type == 'multiclass':
            label_file = f"y_train_multiclass.csv"
        start = time.time()
        data = pd.read_csv(data_set_file)
        labels = pd.read_csv(label_file)
        data = pd.concat([data, labels['label']], axis=1)
        end = time.time()
        logger.info(f'Done! Time elapsed: {(end - start):0.02f}s.')

        if not check_data(data, labels):
            print(f'Train data and label shape not match: {data.shape[0], labels.shape[0]}')
            exit(-1)

        if NEED_DISCRETIZE:
            logger.info('Discretizing the data...')
            data2 = discretizer(data.copy())
            logger.info('Done!')
        else:
            data2 = data.copy()

        logger.info('Start calculating information gain of all columns...')
        start = time.time()
        ig = get_information_gain(data2, 'label', data2.shape[0])
        end = time.time()
        logger.info(f'Done! Time elapsed: {(end - start):0.02f}s.')

        logger.info(f'The information gains{" (discretized) " if NEED_DISCRETIZE else " "}are: {ig}')

        logger.info('Start Calculating Information gain ratio...')
        igr = ig.copy()
        start = time.time()
        for key in igr.keys():
            igr[key] /= entropy(data2, key, data.shape[0])
        end = time.time()
        logger.info(f'Info_gain_ratio{" (discretized) " if NEED_DISCRETIZE else " "}: {igr}, ')
        logger.info(f'Time elaplsed: {(end - start) : 0.02f}s')

        logger.info('Combine results...')
        ig_matrix = pd.concat([pd.DataFrame(ig, index=['Info Gain']), pd.DataFrame(igr, index=['Info Gain Ratio'])]).T
        ig_mean = ig_matrix['Info Gain'].mean()
        result = ig_matrix[ig_matrix['Info Gain'] > ig_mean].sort_values(by='Info Gain Ratio',
                                                                         ascending=False).iloc[:10, :]
        logger.info(f'result:\n{result}')
        new_columns = result.index.tolist()
        logger.info('Creating new data set...')
        new_data = data[new_columns]
        new_data.to_csv(f"X_new_{delta}_bin.csv", index=False)
