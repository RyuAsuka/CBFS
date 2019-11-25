"""
The implementation of feature clustering algorithm.
"""

import pandas as pd
import numpy as np
import time
from my_logger import set_logger

logger = set_logger('clustering')


def is_in_cluster(f, all_clusters):
    """
    To judge whether a feature belongs to a cluster

    Parameters
    ----------
        f : str
            The feature used for searching.
        all_clusters: list[set]
            A list of sets contains all clusters which have already generated.

    Returns
    -------
        int or None:
            If the feature f is found in any cluster, return the cluster's number.
            Else return None.
    """
    for cluster in all_clusters:
        if f in cluster:
            cluster_number = all_clusters.index(cluster)
            return cluster_number
    return None


def compare_and_join(data, f, all_clusters, delta):
    """
    To compare the feature with all features in all clusters. If the distance of feature f and all features in a
    clusters greater than threshold delta, the feature f will join in the cluster. If all maximum distances are less
    than threshold delta, the feature will be composed to a new cluster.

    Parameters
    ----------
        data: pandas.DataFrame
            The data set.
        f: str
            The feature waiting for joining in the cluster.
        all_clusters: list[set]
            The list of all clusters which have already generated.
        delta: float
            The threshold of distance. Suggest to be set between 0.8 and 1.

    Returns
    -------
        int or None:
            If the feature f finally joins in a cluster, return the cluster number.
            Else return None.

    """
    all_max_distances = []
    for cluster in all_clusters:
        max_distance_in_current_cluster = max([np.abs(data[f].corr(data[f_0])) for f_0 in cluster])
        all_max_distances.append(max_distance_in_current_cluster)
    if len(all_max_distances) == 0:
        return None
    if max(all_max_distances) > delta:
        cluster_with_max_distance = np.argmax(all_max_distances)
        logger.debug('Feature {} has max distance {} with clusters {}'.format(f, max(all_max_distances),
                                                                              cluster_with_max_distance))
        return cluster_with_max_distance
    else:
        return None


def build_cluster(f):
    """
    To build a new cluster with feature f.

    Parameters
    ----------
        f: str
            The feature which cannot join in any currently existing cluster.

    Returns
    -------
        set:
            A set which contains only the features f.
    """
    new_cluster = set()
    new_cluster.add(f)
    return new_cluster


def feature_clustering(data, features, delta):
    """
    The core of feature clustering algorithm.

    Parameters
    ----------
        data : pandas.DataFrame
            The data set.
        features : Union[list[str], pandas.Series]
            The feature set.
        delta : float
            The distance threshold.

    Returns
    -------
        list[set] :
            The list of all feature clusters saved as sets.
    """
    all_clusters = []
    for i in range(len(features)):
        feature_in_cluster = is_in_cluster(features[i], all_clusters)
        if feature_in_cluster is None:
            cluster_to_join = compare_and_join(data, features[i], all_clusters, delta)
            if cluster_to_join is None:
                logger.debug('Feature {} cannot be found in any cluster, build a new cluster.'.format(features[i]))
                all_clusters.append(build_cluster(features[i]))
            else:
                # assert isinstance(cluster_to_join, np.int64)
                logger.debug('Feature {} ready to join cluster {},'.format(features[i], cluster_to_join))
                all_clusters[cluster_to_join].add(features[i])
                logger.debug('now cluster {} is {}.'.format(cluster_to_join, all_clusters[cluster_to_join]))
        else:
            logger.debug('Feature {} is in cluster {},'.format(features[i], feature_in_cluster))
            logger.debug('which is {}'.format(all_clusters[feature_in_cluster]))
    return all_clusters


def find_cluster_center(data, cluster_set):
    """
    To find the cluster center of each cluster

    Parameters
    ----------
        data : pandas.DataFrame
            The data set.
        cluster_set : list[set]

    Returns
    -------
        list[str] :
            The list of selected features.
    """
    new_cluster_set = []
    for cluster in cluster_set:
        cluster = list(cluster)
        if len(cluster) == 1 or len(cluster) == 2:
            logger.debug('Feature {} is center of this cluster.'.format(cluster[0]))
            new_cluster_set.append(cluster[0])
        elif len(cluster) > 2:
            average_distance = []
            for i in range(len(cluster)):
                distance_i = []
                for j in range(len(cluster)):
                    if i == j:
                        continue
                    else:
                        distance_fi_fj = np.abs(data[cluster[i]].corr(data[cluster[j]]))
                        distance_i.append(distance_fi_fj)
                average_distance.append(np.mean(distance_i))
            center = int(np.argmax(average_distance))
            logger.debug('Feature {} is center of this cluster.'.format(cluster[center]))
            new_cluster_set.append(cluster[center])
    return new_cluster_set


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
    logger.info(f'Done! Time elapsed: {(end-start):0.02f} s')
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
    logger.info(f"Time elapsed: {(end-start):0.02f}s")
    cluster_centers.append('Label')

    # Save the data set
    logger.info('Saving the data set...')
    start = time.time()
    df = df[cluster_centers]
    df.to_csv('E:\\data\\result\\dataset_random_split1_clustered.csv', index=False)
    end = time.time()
    logger.info(f'Done! Time elapsed: {(end-start):0.02f}s')
    # Information gain ranking algorithm
