import pandas as pd
import numpy as np
import seaborn as sns
import time
import logging
from tqdm import tqdm
from my_logger import set_logger

logger = set_logger('ranking')

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


def get_information_gain(df, nrows):
    """
    Calculate the information gain of each feature.

    Parameter
    ---------
        df : pandas.DataFrame
            The data set organized by pandas.DataFrame.
        nrows : int
            The number of rows of the data set.

    Returns
    -------
        dict[str:float] :
            The information gain of each feature. The key is feature name and the value is its information gain.
    """
    info_gains = {}
    ent_label = entropy(df, 'Label', nrows)
    # for column in tqdm(df.columns[:-1], total=len(df.columns) - 1):
    for column in df.columns[:-1]:
        groups = df.groupby(column)
        column_prob = df[column].value_counts().astype('float64') / nrows
        tablen = column_prob.shape[0]
        cond_ent = 0.0
        # for name, group in tqdm(groups, total=tablen):
        for name, group in groups:
            group_nrows = group.shape[0]
            prob_label = group['Label'].value_counts().astype('float64') / group_nrows
            logged_prob_label = prob_label * np.log2(prob_label)
            try:
                ent = column_prob[name] * logged_prob_label.sum()
            except KeyError:
                ent = 0.0
            cond_ent -= ent
        info_gains[column] = ent_label - cond_ent
    return info_gains


def discretizer(dataframe, bins):
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
        dataframe[col] = pd.cut(dataframe[col], bins=bins, labels=False)
    return dataframe


if __name__ == '__main__':
    time_usage = []
    for delta in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        time_usage_delta = []
        for i in range(1, 11):
            logger.info(f'Reading data set {i}, delta={delta}...')
            data_set_file = f'E:\\data\\result\\clustering_result\\dataset_random_split{i}_clustered_{delta}.csv'
            start = time.time()
            data = pd.read_csv(data_set_file)
            end = time.time()
            logger.info(f'Done! Time elapsed: {(end - start):0.02f}s.')

            NEED_DISCRETIZE = True
            if NEED_DISCRETIZE:
                logger.info('Discretizing the data...')
                data2 = discretizer(data.copy(), bins=100)
                logger.info('Done!')
            else:
                data2 = data.copy()

            logger.info('Start calculating information gain of all columns...')
            start = time.time()
            ig = get_information_gain(data2, data2.shape[0])
            #  = time.time()
            # logger.info(f'Done! Time elapsed: {(end - start):0.02f}s.')

            logger.info(f'The information gains{" (discretized) " if NEED_DISCRETIZE else " "}are: {ig}')

            logger.info('Start Calculating Information gain ratio...')
            igr = ig.copy()
            # start = time.time()
            for key in igr.keys():
                igr[key] /= entropy(data2, key, data.shape[0])
            end = time.time()
            logger.info(f'Info_gain_ratio{" (discretized) " if NEED_DISCRETIZE else " "}: {igr}, ')
            logger.info(f'Time elaplsed: {(end - start) : 0.02f}s')
            time_usage_delta.append(end - start)

            logger.info('Combine results...')
            ig_matrix = pd.concat(
                [pd.DataFrame(ig, index=['Info Gain']), pd.DataFrame(igr, index=['Info Gain Ratio'])]).T
            ig_mean = ig_matrix['Info Gain'].mean()
            result = ig_matrix[ig_matrix['Info Gain'] > ig_mean].sort_values(by='Info Gain Ratio',
                                                                             ascending=False).iloc[:10, :]
            logger.info(f'result:\n{result}')
            new_columns = result.index.tolist()
            new_columns.append('Label')
        time_usage.append(time_usage_delta)
    print(time_usage)
    # logger.info('Creating new data set...')
    # new_data = data[new_columns]
    # new_data.to_csv(f'E:\\data\\result\\new_data_sets\\new_dataset_{i}_{delta}.csv', index=False)
