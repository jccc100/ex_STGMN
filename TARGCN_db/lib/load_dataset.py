import os
import numpy as np

def load_st_dataset(dataset):
    #output B, N, D
    if dataset == 'PEMSD4':
        data_path = os.path.join('../data/PEMSD4/pems04.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMSD8':
        data_path = os.path.join('../data/PEMSD8/pems08.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'wujing_5':
        data_path = os.path.join('../data/wujing/wujing_5.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'wujing_5_speed':
        data_path = os.path.join('../data/wujing/wujing_5_speed.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'wujing_10':
        data_path = os.path.join('../data/wujing/wujing_10.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'wujing_10_speed':
        data_path = os.path.join('../data/wujing/wujing_10_speed.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    elif dataset == 'PEMS03':
        data_path = os.path.join('../data/PEMS03/PEMS03w.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                if data[i,j]<=10:
                    data[i,j]=0
    elif dataset == 'PEMSD7':
        data_path = os.path.join('../data/PEMSD7/PEMS07.npz')
        data = np.load(data_path)['data'][:, :, 0]  #onley the first dimension, traffic flow data
    else:
        raise ValueError
    if len(data.shape) == 2:
        data = np.expand_dims(data, axis=-1)
    print('Load %s Dataset shaped: ' % dataset, data.shape, data.max(), data.min(), data.mean(), np.median(data))
    return data
