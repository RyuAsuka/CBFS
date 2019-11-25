import pandas as pd
import numpy as np
from tqdm import tqdm


pd.set_option('display.max_rows', None)
print('Get data shape')
data = pd.read_csv('E:\\data\\result\\dataset_random_split1.csv', low_memory=False)
total_size = data.shape[0]
chunksize = 1000
print(f'total_size={total_size}')
del data
print(f'Read data by {chunksize} rows per chunk')
data = pd.read_csv('E:\\data\\result\\dataset_random_split1.csv', chunksize=chunksize)
iter_num = int(np.ceil(total_size / chunksize))
print(f'{iter_num} iters')

i = 0
for chunk in tqdm(data, total=iter_num):
    dtype = chunk['Dst Port'].dtypes
    if str(dtype) != 'int64':
        tqdm.write(f'{i}: {chunk["Dst Port"]}')
    i += 1
