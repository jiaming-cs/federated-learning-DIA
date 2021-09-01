from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

def fit_kmeans(data_x, data_y):
    data_x_flatten = [x.flatten() for x in data_x]
    label_num = np.unique(data_y).shape[0]
    
    kmeans = KMeans(n_clusters=label_num).fit(data_x_flatten)
    pred_y = kmeans.labels_
    # correct_num = sum([1 for pred, gt in zip(pred_y, data_y) if pred == gt])
    # acc = correct_num / label_num
    # print(f'acc: {acc}')
    return [pred for pred, gt in zip(pred_y, data_y) if pred == gt] # Return indecies of kmeans verified data
    