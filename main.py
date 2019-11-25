import logging
import time
import pandas as pd
from FeatureClustering import feature_clustering
from FeatureClustering import find_cluster_center
from InformationGainRanking import get_information_gain
from InformationGainRanking import discretizer

# Configure logger
logger = logging.getLogger('main')
logger.setLevel(logging.INFO)
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='[%(asctime)s][%(levelname)s] %(message)s')
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

WORK_DIR = 'E:\\data\\result\\'
DATA_FILES = ['']


if __name__ == '__main__':

    # Reading data and preprocessing
    logger.info('Reading data...')
    start = time.time()
    input_data = 'E:\\data\\result\\dataset_random_split1_modified.csv'
    df = pd.read_csv(input_data)
    logger.info(f'Done! time elapsed: {(time.time() - start):0.02f} s')
    logger.info(f'Dataset has {(len(df.columns) - 1):d} features.')

    # Feature Clustering
    columns = df.columns[:-1]
    logger.info('Running Feature Clustering...')
    start = time.time()
    C = feature_clustering(df, columns, 0.9)
    end = time.time()
    logger.info(f'Done! Time elapsed: {(end - start):0.02f} s')
    logger.info('The clustered features are:')
    for c in C:
        logger.info(f'{c}')
    logger.info(f"The cluster set have {len(C):d} clusters.")
    logger.info(f"All clusters have {sum([len(x) for x in C]):d} features")

    # Find cluster centers
    logger.info('Running finding cluster center algorithm...')
    start = time.time()
    cluster_centers = find_cluster_center(df, C)
    end = time.time()
    logger.info(f"Done! The clustered centers are: {cluster_centers}")
    logger.info(f"Time elapsed: {(end - start):0.02f}s")
    cluster_centers.append('Label')

    # Save the data set
    logger.info('Saving the data set...')
    start = time.time()
    df = df[cluster_centers]
    df.to_csv('E:\\data\\result\\dataset_random_split1_clustered.csv', index=False)
    end = time.time()
    logger.info(f'Done! Time elapsed: {(end - start):0.02f}s')

    # Information gain ranking algorithm