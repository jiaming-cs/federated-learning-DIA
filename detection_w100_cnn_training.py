#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 17:25:46 2020

@author: aaronli
"""

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
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier
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
LR = 0.001   # learning rate
KFOLD = 5
isGPU = torch.cuda.is_available()

# detection deep learning dataset path
data_path_DL = '../new_clean_data/w100_final_dataset/fault_detection/deep_learning/'

all_data_DL = loader.waveformDetectionDL(data_path_DL)

#data_path_DL = '../new_clean_data/w100_final_dataset/fault_diagnosis/deep_learning/'
#all_data_DL = loader.waveformDiagnosisDL(data_path_DL)


#-------------------create the CNN Net ----------------------------------------
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ), # -> (16, 53, 3)
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, ceil_mode=False),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), # -> (16, 25, 6)
            nn.ReLU(),
            nn.MaxPool2d(2),
        ) # -> (32, 13, 3)
        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), # -> (16, 25, 6)
            nn.ReLU(),
            nn.MaxPool2d(2),
        ) # -> (32, 13, 3)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), # -> (16, 25, 6)
            nn.ReLU(),
            nn.MaxPool2d(2),
        ) # -> (32, 13, 3)
        self.conv4 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2), # -> (16, 25, 6)
            nn.ReLU(),
            nn.MaxPool2d(2),
        ) 
        self.out = nn.Linear(32 * 25 * 1, 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        output = self.out(x)
        return output


data_output = {'cnn':{'F1':[], 'precision':[], 'recall':[], 'accuracy':[], 'auc':[], 'fpr':[], 'tpr':[], 'test_loss':[], 'train_loss':[]}}

   
#%%   
for num_of_training in range(KFOLD):
    print('------------------fold {}------------------------'.format(num_of_training + 1))
    cnn = CNN()
    
    if isGPU:
        cnn = nn.DataParallel(cnn, device_ids=[0,1])
        cnn.cuda()

    cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
    loss_func = nn.CrossEntropyLoss()
    
     # print the structure of the network
    print(cnn)
    
     # data partition: 15% testing, 85% training
    training_data, test_data = torch.utils.data.random_split(all_data_DL, [int(all_data_DL.len * 0.85), all_data_DL.len - int(all_data_DL.len * 0.85)])
    training_Loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
    test_Loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
    
    # training and testing
    cnn_test_loss_draw = []
    cnn_loss_draw = []
    
    for epoch in range(EPOCH):
        print('-----------------------------epoch {}---------------------------'.format(epoch + 1))
        
        # training-----------------------------------------
        cnn_train_loss = 0.
        cnn_train_acc = 0.
        cnn.train()
        
        for step, (batch_x, batch_y) in enumerate(training_Loader):
            batch_x_cnn = torch.unsqueeze(batch_x, dim=1).type(torch.float)
            
            if isGPU:
                batch_x_cnn = batch_x_cnn.cuda()
                batch_y = batch_y.cuda()
                
            output_cnn = cnn(batch_x_cnn)
            loss_cnn = loss_func(output_cnn, batch_y)
            cnn_train_loss += loss_cnn.item()

            if isGPU:
                cnn_pred = torch.max(output_cnn, 1)[1].cuda()
            else:
                cnn_pred = torch.max(output_cnn, 1)[1]  

            cnn_train_correct = (cnn_pred == batch_y).sum()
            
            cnn_train_acc += cnn_train_correct.item()
            
            cnn_optimizer.zero_grad()
            
            loss_cnn.backward()
            
            cnn_optimizer.step()

        print('CNN:\n Train Loss: {:.6f}, Training Accuracy: {:.6f}\n'.format(cnn_train_loss / 
              (len(training_data)), cnn_train_acc / (len(training_data))))

        cnn_loss_draw.append(cnn_train_loss/(len(training_data)))
        
        
        # evaluation--------------------------------------------------
        cnn.eval()
        
        cnn_eval_loss = 0.        
        cnn_eval_acc = 0.
        
        cnn_final_prediction = np.array([])
        cnn_final_test = np.array([])
        cnn_f1_score = []
        cnn_recall = []
        cnn_precision = []
                
        for step, (batch_x, batch_y) in enumerate(test_Loader):
            batch_x_cnn = torch.unsqueeze(batch_x, dim=1).type(torch.float)
            
            if isGPU:
                batch_x_cnn = batch_x_cnn.cuda()
                batch_y = batch_y.cuda()
    
            output_cnn = cnn(batch_x_cnn)

            loss_cnn = loss_func(output_cnn, batch_y)

            cnn_eval_loss += loss_cnn.item()
            
            cnn_pred = torch.max(output_cnn, 1)[1]

            cnn_train_correct = (cnn_pred == batch_y).sum()
            
            if isGPU:
                cnn_pred = torch.max(output_cnn, 1)[1].cuda()
            else:
                cnn_pred = torch.max(output_cnn, 1)[1]

            cnn_eval_acc += cnn_train_correct.item()
            
            # F1 metrics
            cnn_final_prediction = np.concatenate((cnn_final_prediction, cnn_pred.cpu().numpy()), axis=0)
            cnn_final_test = np.concatenate((cnn_final_test, batch_y), axis=0)
        
        cnn_f1_score.append(sklearn.metrics.f1_score(cnn_final_test, cnn_final_prediction, average='binary').item())

        cnn_recall.append(sklearn.metrics.recall_score(cnn_final_test, cnn_final_prediction, average='binary').item())

        cnn_precision.append(sklearn.metrics.precision_score(cnn_final_test, cnn_final_prediction, average='binary').item())
        
        print('CNN:\n Test Loss: {:.6f}, Test Accuracy: {:.6f}'.format(cnn_eval_loss / 
              (len(test_data)), cnn_eval_acc / (len(test_data))))

        cnn_test_loss_draw.append(cnn_eval_loss/(len(test_data)))
        
        print('CNN:\n F1: {}, recall: {}, precision: {}'.format(cnn_f1_score[-1], cnn_recall[-1], cnn_precision[-1]))
        
    # ROC curve and AUC

    cnn_test_y = label_binarize(cnn_final_test, classes=[0, 1])
    cnn_pred_y = label_binarize(cnn_final_prediction, classes=[0, 1])
    
    #cnn_test_y = label_binarize(cnn_final_test, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    #cnn_pred_y = label_binarize(cnn_final_prediction, classes=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    cnn_fpr, cnn_tpr, _ = roc_curve(cnn_test_y.ravel(), cnn_pred_y.ravel())
    cnn_roc_auc = auc(cnn_fpr, cnn_tpr)
        
    data_output['cnn']['F1'].append(cnn_f1_score[-1])
    data_output['cnn']['precision'].append(cnn_precision[-1])
    data_output['cnn']['recall'].append(cnn_recall[-1])
    data_output['cnn']['accuracy'].append(cnn_eval_acc / (len(test_data)))
    data_output['cnn']['auc'].append(cnn_roc_auc.item())
    data_output['cnn']['fpr'].append(list(cnn_fpr))
    data_output['cnn']['tpr'].append(list(cnn_tpr))
    data_output['cnn']['test_loss'].append(cnn_test_loss_draw)
    data_output['cnn']['train_loss'].append(cnn_loss_draw) 

#%% ---------------------dataoutput------------------------------------------------------------------

for i in range(KFOLD):
    print('--------Fold {}----------------'.format(i))

   
    print('CNN: F1:{}, prec:{}, rec:{}, acc:{}, auc:{}'.format(data_output['cnn']['F1'][i],
          data_output['cnn']['precision'][i], data_output['cnn']['recall'][i],
          data_output['cnn']['accuracy'][i], data_output['cnn']['auc'][i]))
    
    # save figures for ann, cnn, lstm
    
    plt.figure()
    
    plt.plot(data_output['cnn']['test_loss'][i], label='CNN testing')
    plt.plot(data_output['cnn']['train_loss'][i], label='CNN training')
   
    plt.legend()
    
    plt.xlabel('Epoch')
    plt.xticks(np.arange(0, EPOCH+1, EPOCH/10))
    plt.ylabel('Loss')
    plt.title('Loss Function')
    plt.savefig('../fig/w100_detection/'+'Loss_Kfold_CNN_w100_detection_'+str(i+1)+'.png',dpi=500)
    
    # save figures of ROC curve
    plt.figure()
    plt.plot(data_output['cnn']['fpr'][i], data_output['cnn']['tpr'][i], label='CNN (AUC = {0:0.2f})'.format(data_output['cnn']['auc'][i]))

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.legend()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve')
    plt.savefig('../fig/w100_detection/'+'ROC_Kfold_CNN_w100_detection_'+str(i+1)+'.png',dpi=500)
    
    
pickle_out = open('CNN_w100_10fold_detection.pickle', 'wb')
pickle.dump(data_output, pickle_out)
pickle_out.close()

# end time
end_time = datetime.datetime.now()
print('Running time: {}'.format(end_time - start_time))

