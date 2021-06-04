#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 22:15:25 2020

@author: aaronli
"""

import loader
import torch
from torch import nn
# import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import sklearn.metrics
from sklearn.metrics import roc_curve, auc, confusion_matrix
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
EPOCH = 300
BATCH_SIZE = 128
TIME_STEP = 100  # length of LSTM time sequence, or window size
INPUT_SIZE = 6 # num of feature for deep learning
LR = 0.002   # learning rate
KFOLD = 10
isGPU = torch.cuda.is_available()

# detection deep learning dataset path
# data_path_FT = '../new_clean_data/final_dataset/fault_detection/feature/'

# all_data_FT = loader.waveformDetectionFeature(data_path_FT)

data_path_FT = '../new_clean_data/w100_final_dataset/fault_diagnosis/feature/'


all_data_FT = loader.waveformDiagnosisFeature(data_path_FT)


#-------------------create the NN Net -----------------------------------------
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(18, 128),
            nn.ReLU(),
            nn.Linear(128,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,9),
#             nn.ReLU(),
#             nn.Linear(64,32),
#             nn.ReLU(),
#             nn.Linear(32,2),
        )
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output

data_output = {'ann':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'test_loss':[], 'train_loss':[], 'confusion_mat':[]}}

    
#%% for ANN training
for num_of_training in range(KFOLD):
    print('------------------fold {}------------------------'.format(num_of_training + 1))
    ann = ANN()
    if isGPU:
        ann.cuda()
        ann = nn.DataParallel(ann, device_ids=[0, 1, 2, 3])
#     ann_optimizer = torch.optim.Adam(ann.parameters(), lr=LR)
    ann_optimizer = torch.optim.SGD(ann.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
    print(ann)
    
#     #scale X for normalization
#     min_max_scaler = preprocessing.MinMaxScaler()
#     all_data_FT.x_data = min_max_scaler.fit_transform(all_data_FT.x_data.numpy())
#     print(all_data_FT.x_data[0])
    
    training_data, test_data = torch.utils.data.random_split(all_data_FT, [int(all_data_FT.len * 0.85), all_data_FT.len - int(all_data_FT.len * 0.85)])
    training_Loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_Loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    ann_test_loss_draw = []
    ann_loss_draw = []
    
    for epoch in range(EPOCH):
        print('-----------------------epoch {}-------------------------'.format(epoch + 1))
         # training-----------------------------------------
        ann_train_loss = 0.
        ann_train_acc = 0.
        
        ann.train()

        for step, (batch_x, batch_y) in enumerate(training_Loader):
            batch_x = batch_x.view(-1, 1, 18)
            batch_x = batch_x.float()
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
                
            output_ann = ann(batch_x)

            loss_ann = loss_func(output_ann, batch_y)
            
            ann_train_loss += loss_ann.item()
            
            if isGPU:
                ann_pred = torch.max(output_ann, 1)[1].cuda()
            else:
                ann_pred = torch.max(output_ann, 1)[1]
        
            ann_train_correct = (ann_pred == batch_y).sum()
            ann_train_acc += ann_train_correct.item()
            ann_optimizer.zero_grad()
            loss_ann.backward()
            ann_optimizer.step()
            
        print('ANN:\n Train Loss: {:.6f}, Training Accuracy: {:.6f}\n'.format(ann_train_loss / 
              (len(training_data)), ann_train_acc / (len(training_data))))

        ann_loss_draw.append(ann_train_loss/(len(training_data)))
        
        # evaluation----------------------------------------------------------
        ann.eval()
        ann_eval_loss = 0.
        ann_eval_acc = 0.
        ann_final_prediction = np.array([])
        ann_final_test = np.array([])
        ann_f1_score = []
        ann_recall = []
        ann_precision = []
        
        for step, (batch_x, batch_y) in enumerate(test_Loader):
            batch_x = batch_x.view(-1, 1, 18)
            batch_x = batch_x.float()
            
            if isGPU:
                batch_x = batch_x.cuda()
                batch_y = batch_y.cuda()
    
            output_ann = ann(batch_x)
            loss_ann = loss_func(output_ann, batch_y)
            ann_eval_loss += loss_ann.item()
            ann_pred = torch.max(output_ann, 1)[1]
            ann_train_correct = (ann_pred == batch_y).sum()
            
            if isGPU:
                ann_pred = torch.max(output_ann, 1)[1].cuda()
            else:
                ann_pred = torch.max(output_ann, 1)[1]

            ann_eval_acc += ann_train_correct.item()
            
            
            # F1 metrics
            
            ann_final_prediction = np.concatenate((ann_final_prediction, ann_pred.cpu().numpy()), axis=0)
            ann_final_test = np.concatenate((ann_final_test, batch_y), axis=0)
            
        #ann_f1_score.append(sklearn.metrics.f1_score(ann_final_test, ann_final_prediction, average='weighted').item())
        #ann_recall.append(sklearn.metrics.recall_score(ann_final_test, ann_final_prediction, average='macro').item())
        #ann_precision.append(sklearn.metrics.precision_score(ann_final_test, ann_final_prediction, average='weighted', zero_division='warn').item())
        
        current_f1 = sklearn.metrics.f1_score(ann_final_test, ann_final_prediction, average='weighted').item()
        current_precision = sklearn.metrics.precision_score(ann_final_test, ann_final_prediction, average='weighted', zero_division=1).item()
        ann_f1_score.append(current_f1)
        ann_precision.append(current_precision)
        ann_recall.append((current_f1*current_precision) / (2*current_precision - current_f1))

        print('ANN:\n Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(ann_eval_loss / 
            (len(test_data)), ann_eval_acc / (len(test_data))))
        
        ann_test_loss_draw.append(ann_eval_loss/(len(test_data)))
        print('ANN:\n F1: {}, recall: {}, precision: {}'.format(ann_f1_score[-1], ann_recall[-1], ann_precision[-1]))
    
    
    
    
    # confusing matrix
    current_cm = confusion_matrix(list(ann_final_test), list(ann_final_prediction))
    print(current_cm)
    
    # ROC curve and AUC
    ann_test_y = label_binarize(ann_final_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ann_pred_y = label_binarize(ann_final_prediction, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    ann_fpr, ann_tpr, _ = roc_curve(ann_test_y.ravel(), ann_pred_y.ravel())
    ann_roc_auc = auc(ann_fpr, ann_tpr)
    
    data_output['ann']['F1'].append(ann_f1_score[-1])
    data_output['ann']['precision'].append(ann_precision[-1])
    data_output['ann']['recall'].append(ann_recall[-1])
    data_output['ann']['accuracy'].append(ann_eval_acc / (len(test_data)))
    data_output['ann']['auc'].append(ann_roc_auc.item())
    data_output['ann']['fpr'].append(list(ann_fpr))
    data_output['ann']['tpr'].append(list(ann_tpr))
    data_output['ann']['test_loss'].append(ann_test_loss_draw)
    data_output['ann']['train_loss'].append(ann_loss_draw)    
    
    data_output['ann']['confusion_mat'].append(current_cm)
        
        
    
#%% ---------------------dataoutput------------------------------------------------------------------

for i in range(KFOLD):
    print('--------Fold {}----------------'.format(i))
    print('ANN: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['ann']['F1'][i],
          data_output['ann']['precision'][i], data_output['ann']['recall'][i],
          data_output['ann']['accuracy'][i], data_output['ann']['auc'][i]))
    
    # save figures for ann, cnn, lstm
    
    plt.figure()

    plt.plot(data_output['ann']['test_loss'][i], label='ANN testing')
    plt.plot(data_output['ann']['train_loss'][i], label='ANN training')
    
    plt.legend()
    
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, EPOCH+1, EPOCH/10))
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.savefig('../fig/w100_detection/'+'Loss_Kfold_ANN_w100_diagnosis_'+str(i+1)+'.png',dpi=500)
    
    # save figures of ROC curve
    plt.figure()
    plt.plot(data_output['ann']['fpr'][i], data_output['ann']['tpr'][i], label='ANN (AUC = {0:0.2f})'.format(data_output['ann']['auc'][i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig('../fig/w100_detection/'+'ROC_Kfold_ANN_w100_diagnosis_'+str(i+1)+'.png',dpi=500)
    
    
pickle_out = open('Ann_w100_10fold_diagnosis.pickle', 'wb')
pickle.dump(data_output, pickle_out)
pickle_out.close()

# end time
end_time = datetime.datetime.now()
print('Running time: {}'.format(end_time - start_time))