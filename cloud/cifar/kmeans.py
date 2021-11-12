from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

def fit_kmeans(data_x, data_y):
    
    print('x_data: ', data_x.shape)
    print('y_data: ', data_y.shape)
    data_y = data_y.flatten()
    data_x_flatten = np.asarray([x.numpy().flatten() for x in data_x])
    # data_x_flatten = data_x.numpy()
    print(data_x_flatten.shape)
    label_num = np.unique(data_y).shape[0]
    kmeans = KMeans(n_clusters=label_num).fit(data_x_flatten)
    pred_y = kmeans.labels_
    mapping = [(pred, assigned) for pred, assigned in zip(pred_y, data_y)]
    
    mapping_counter = {}
    for m in mapping:
        mapping_counter[m] = mapping_counter.get(m, 0) + 1
    mapping_counter_list = [(k, v) for k, v in mapping_counter.items()]
    mapping_counter_list.sort(key = lambda m: -m[1])
    for i, pair in enumerate(mapping_counter_list):
        print(f'{i}-{pair}')
    most_common_mapping = {}
    for m, _ in mapping_counter_list:
        if m[0] not in most_common_mapping:
            most_common_mapping[m[0]] = m[1]
    
    pred_y_aligned = [most_common_mapping[i] for i in pred_y]
        
    
    
    return [i for i, (pred, gt) in enumerate(zip(pred_y_aligned, data_y)) if pred == gt] # Return indecies of kmeans verified data
    