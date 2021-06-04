#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:15:25 2020

@author: aaronli
"""

import loader
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import pickle
import numpy as np
import datetime


# Running time calculate: start time
start_time = datetime.datetime.now()

# Hyper Parameters
EPOCH = 20
BATCH_SIZE = 32
TIME_STEP = 100  # length of LSTM time sequence, or window size
INPUT_SIZE = 6 # num of feature for deep learning
LR = 0.01   # learning rate
KFOLD = 1
isGPU = torch.cuda.is_available()

# detection deep learning dataset path
data_path_FT = '../new_clean_data/w50_final_dataset/fault_diagnosis/feature/'

all_data_FT = loader.waveformDiagnosisFeature(data_path_FT)

data_output = {'knn':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'confusion_mat':[]}, 
               'dtree':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'confusion_mat':[]}, 
               'svm':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'confusion_mat':[]}}

for num_of_training in range(KFOLD):
    
    # machine learning methods
    ml_X = all_data_FT.x_data.numpy()
    ml_X = ml_X.reshape(ml_X.shape[0], -1)
    ml_Y = all_data_FT.y_data.numpy()
    
    # scale X for normalization
#     min_max_scaler = preprocessing.MinMaxScaler()
#     ml_X = min_max_scaler.fit_transform(ml_X)
#     print(ml_Y)
#     print(ml_X[0], ml_X[1])
    
    ml_X_train, ml_X_test, ml_Y_train, ml_Y_test = train_test_split(ml_X, ml_Y, test_size=0.15)
    
#     # train dtree
    dtree_model = DecisionTreeClassifier(max_depth = 10).fit(ml_X_train, ml_Y_train)
    dtree_pred = dtree_model.predict(ml_X_test)
    
    # train SVM 
    svm_model = SVC(kernel = 'rbf', C = 1, max_iter = -1).fit(ml_X_train, ml_Y_train)
    svm_pred = svm_model.predict(ml_X_test)
    
    svm_pred_training = svm_model.predict(ml_X_train)
#     # train KNN 
    knn_model = KNeighborsClassifier(n_neighbors = 17).fit(ml_X_train, ml_Y_train)
    knn_pred = knn_model.predict(ml_X_test)
    
#     # metrics for machine learning methods
    dtree_f1 = sklearn.metrics.f1_score(ml_Y_test, dtree_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    dtree_precision = sklearn.metrics.precision_score(ml_Y_test, dtree_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    #dtree_recall = sklearn.metrics.recall_score(ml_Y_test, dtree_pred, average='weighted')
    #dtree_recall = ((dtree_f1*dtree_precision) / (2*dtree_precision - dtree_f1))
    dtree_recall = sklearn.metrics.recall_score(ml_Y_test, dtree_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    dtree_acc = sklearn.metrics.accuracy_score(ml_Y_test, dtree_pred)
    
    svm_f1 = sklearn.metrics.f1_score(ml_Y_test, svm_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    svm_precision = sklearn.metrics.precision_score(ml_Y_test, svm_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    svm_recall = sklearn.metrics.recall_score(ml_Y_test, svm_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    #svm_recall = ((svm_f1*svm_precision) / (2*svm_precision - svm_f1))
    svm_acc = sklearn.metrics.accuracy_score(ml_Y_test, svm_pred)
    
    knn_f1 = sklearn.metrics.f1_score(ml_Y_test, knn_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    knn_precision = sklearn.metrics.precision_score(ml_Y_test, knn_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    knn_recall = sklearn.metrics.recall_score(ml_Y_test, knn_pred, labels=[1,2,3,4,5,6,7,8], average='micro')
    #knn_recall = ((knn_f1*knn_precision) / (2*knn_precision - knn_f1))
    knn_acc = sklearn.metrics.accuracy_score(ml_Y_test, knn_pred)

    # confusion matrix
    dtree_current_cm = confusion_matrix(ml_Y_test, dtree_pred)
    svm_current_cm = confusion_matrix(ml_Y_test, svm_pred)
    knn_current_cm = confusion_matrix(ml_Y_test, knn_pred)
    
    print("dtree_matrix:\n{}\n".format(dtree_current_cm))
    print("svm_matrix:\n{}\n".format(svm_current_cm))
    print("knn_matrix:\n{}\n".format(knn_current_cm))
    
#     # for ROC curve
    svm_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    svm_pred_y = label_binarize(svm_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    svm_fpr, svm_tpr, _ = roc_curve(svm_test_y.ravel(), svm_pred_y.ravel())
    #svm_roc_auc = sklearn.metrics.auc(svm_fpr, svm_tpr)
    svm_roc_auc = sklearn.metrics.roc_auc_score(svm_test_y, svm_pred_y)
    
    knn_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    knn_pred_y = label_binarize(knn_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    knn_fpr, knn_tpr, _ = roc_curve(knn_test_y.ravel(), knn_pred_y.ravel())
    #knn_roc_auc = sklearn.metrics.auc(knn_fpr, knn_tpr)
    knn_roc_auc = sklearn.metrics.roc_auc_score(knn_test_y, knn_pred_y)
    
    dtree_test_y = label_binarize(ml_Y_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    dtree_pred_y = label_binarize(dtree_pred, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    dtree_fpr, dtree_tpr, _ = roc_curve(dtree_test_y.ravel(), dtree_pred_y.ravel())
    #dtree_roc_auc = sklearn.metrics.auc(dtree_fpr, dtree_tpr)
    dtree_roc_auc = sklearn.metrics.roc_auc_score(dtree_test_y, dtree_pred_y)
    
#     # machine learning method output
    data_output['knn']['F1'].append(knn_f1)
    data_output['knn']['precision'].append(knn_precision)
    data_output['knn']['recall'].append(knn_recall)
    data_output['knn']['accuracy'].append(knn_acc)
    data_output['knn']['auc'].append(knn_roc_auc.item())
    data_output['knn']['fpr'].append(list(knn_fpr))
    data_output['knn']['tpr'].append(list(knn_tpr))
    data_output['knn']['confusion_mat'].append(knn_current_cm)
    
    data_output['dtree']['F1'].append(dtree_f1)
    data_output['dtree']['precision'].append(dtree_precision)
    data_output['dtree']['recall'].append(dtree_recall)
    data_output['dtree']['accuracy'].append(dtree_acc)
    data_output['dtree']['auc'].append(dtree_roc_auc.item())
    data_output['dtree']['fpr'].append(list(dtree_fpr))
    data_output['dtree']['tpr'].append(list(dtree_tpr))
    data_output['dtree']['confusion_mat'].append(dtree_current_cm)
    
    data_output['svm']['F1'].append(svm_f1)
    data_output['svm']['precision'].append(svm_precision)
    data_output['svm']['recall'].append(svm_recall)
    data_output['svm']['accuracy'].append(svm_acc)
    data_output['svm']['auc'].append(svm_roc_auc.item())
    data_output['svm']['fpr'].append(list(svm_fpr))
    data_output['svm']['tpr'].append(list(svm_tpr))
    data_output['svm']['confusion_mat'].append(svm_current_cm)
    
for i in range(KFOLD):
    print('--------Fold {}----------------'.format(i))


    print('KNN: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['knn']['F1'][i],
           data_output['knn']['precision'][i], data_output['knn']['recall'][i],
           data_output['knn']['accuracy'][i], data_output['knn']['auc']))
    print('SVM: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['svm']['F1'][i],
           data_output['svm']['precision'][i], data_output['svm']['recall'][i],
           data_output['svm']['accuracy'][i], data_output['svm']['auc']))
    print('Dtree: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['dtree']['F1'][i],
           data_output['dtree']['precision'][i], data_output['dtree']['recall'][i],
           data_output['dtree']['accuracy'][i], data_output['dtree']['auc']))

pickle_out = open('ML_w50_10fold_diagnosis.pickle', 'wb')
pickle.dump(data_output, pickle_out)
pickle_out.close()            
            
# end time
end_time = datetime.datetime.now()
print('Running time: {}'.format(end_time - start_time))
    