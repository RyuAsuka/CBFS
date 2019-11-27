import time
import pandas as pd
import numpy as np
from tqdm import tqdm
from my_logger import set_logger
from TimeFormatter import time_formatter

logger = set_logger('prepro')


def preprocessing(input_data_file, output_data_file):
    logger.info('Reading data...')
    start = time.time()
    data = pd.read_csv(input_data_file)
    end = time.time()
    logger.info(f'Done! Time elapsed: {time_formatter(end - start)}s')
    # These two columns cannot be used in our feature clustering algorithm
    data.drop(axis=1, columns=['Dst Port', 'Protocol'], inplace=True)

    # Exam error values and fix them
    index_sets = []
    for col in data.columns[:-1]:
        if col == 'Init Fwd Win Byts' or col == 'Init Bwd Win Byts':
            continue
        if data[data[col] < 0].shape[0] > 0:
            logger.debug(col)
            index_sets.append(set(data[data[col] < 0].index.tolist()))
    logger.debug(index_sets)
    for i in range(1, len(index_sets)):
        index_sets[0].union(index_sets[i])
    logger.info(list(index_sets[0]))

    data.drop(axis=0, index=list(index_sets[0]), inplace=True)
    data.at[data[data['Init Fwd Win Byts'] < 0].index, 'Init Fwd Win Byts'] = 0
    data.at[data[data['Init Bwd Win Byts'] < 0].index, 'Init Bwd Win Byts'] = 0
    des = data.describe().T
    data.drop(columns=des[des['std'] == 0].index, inplace=True)
    logger.info('Saving modified data...')
    start = time.time()
    data.to_csv(output_data_file, index=False)
    end = time.time()
    logger.info(f'Done! Time elapsed: {time_formatter(end - start)}s')


if __name__ == '__main__':
    for i in range(1, 5):
        logger.info(f'Preprocessing file {i}')
        preprocessing("E:\\data\\result\\dataset_random_split" + str(i) + ".csv",
                      "E:\\data\\result\\dataset_random_split" + str(i) + "_modified.csv")
