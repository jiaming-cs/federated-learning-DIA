import flwr as fl
import numpy as np
from model import LSTM
from loader import WaveformDetectionDLPickle
from torch.utils.data import DataLoader
import torch
from torch import nn
from collections import OrderedDict
from argparse import ArgumentParser
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score
DATASET_DIR = './'
INPUT_SIZE = 6 # num of feature for deep learning
EPOCH = 300
BATCH_SIZE = 128
TIME_STEP = 100  # length of LSTM time sequence, or window size
VAL_SPLIT = 0.1
LR = 0.001   # learning rate

isGPU = torch.cuda.is_available()
loss_func = nn.CrossEntropyLoss()

history = {'train':{'loss':[], 'acc':[]}, 'val':{'loss':[], 'acc':[]}, 'test':{'loss':[], 'acc':[], 'f1':[], 'recall':[], 'precision':[]}}

def load_data():
    dataset = WaveformDetectionDLPickle(DATASET_DIR, file_name='detection.pkl')
    return dataset

lstm = LSTM()
dataset = load_data()

if isGPU:
    lstm = nn.DataParallel(lstm, device_ids=[0,1,2,3])
    lstm.cuda()

lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

val_num = int(len(dataset) * VAL_SPLIT)

training_data, val_data, test_data = torch.utils.data.random_split(dataset, [len(dataset)-2*val_num, val_num, val_num])
training_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


for epoch in range(EPOCH):
    lstm_train_loss = 0
    
    lstm_train_acc = 0
    
    lstm.train()
    
    for step, (batch_x, batch_y) in enumerate(training_loader):
        batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
        
        if isGPU:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
        
        output_lstm = lstm(batch_x)
        
        batch_y = batch_y.type(torch.LongTensor)
        
        loss_lstm = loss_func(output_lstm, batch_y)
        
        lstm_train_loss += loss_lstm.item()

        if isGPU:
            lstm_pred = torch.max(output_lstm, 1)[1].cuda()
        else:
            lstm_pred = torch.max(output_lstm, 1)[1]
        
        lstm_train_correct = (lstm_pred == batch_y).sum()
        
        lstm_train_acc += lstm_train_correct.item()
        
        lstm_optimizer.zero_grad()
        
        loss_lstm.backward()
        
        lstm_optimizer.step()
    
    lstm_train_acc /= float(len(training_loader.dataset))
    lstm_train_loss /= float(len(training_loader.dataset))
    history['train']['acc'].append(lstm_train_acc)
    history['train']['loss'].append(lstm_train_loss)

    lstm_val_loss = 0
    lstm_val_acc = 0
    for step, (batch_x, batch_y) in enumerate(val_loader):
        batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
        
        if isGPU:
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()

        output_lstm = lstm(batch_x)
        batch_y = batch_y.type(torch.LongTensor)
        loss_lstm = loss_func(output_lstm, batch_y)
        
        lstm_val_loss += loss_lstm.item()
        
        lstm_pred = torch.max(output_lstm, 1)[1]
        
        lstm_val_correct = (lstm_pred == batch_y).sum()
        
        
        if isGPU:
            lstm_pred = torch.max(output_lstm, 1)[1].cuda()
        else:
            lstm_pred = torch.max(output_lstm, 1)[1]
        
        lstm_val_acc += lstm_val_correct.item()
                    
        # F1 metrics

    lstm_val_acc /=  float(len(val_loader.dataset))
    lstm_val_loss /= float(len(val_loader.dataset))
    history['val']['acc'].append(lstm_val_acc)
    history['val']['loss'].append(lstm_val_loss)

    print(f'Epoch:{epoch}')
    print(f'Train Loss: {lstm_train_loss}, Accuracy: {lstm_train_acc}')
    print(f'Validation Loss: {lstm_val_loss}, Accuracy: {lstm_val_acc}')
torch.save(lstm.state_dict(), './model_weights.pth')
lstm.eval()
lstm_eval_loss = 0
lstm_eval_acc = 0
lstm_final_prediction = np.array([])
lstm_final_test = np.array([])
for step, (batch_x, batch_y) in enumerate(test_loader):
    batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
    
    if isGPU:
        batch_x = batch_x.cuda()
        batch_y = batch_y.cuda()

    output_lstm = lstm(batch_x)
    batch_y = batch_y.type(torch.LongTensor)
    loss_lstm = loss_func(output_lstm, batch_y)
    
    lstm_eval_loss += loss_lstm.item()
    
    lstm_pred = torch.max(output_lstm, 1)[1]
    
    lstm_eval_correct = (lstm_pred == batch_y).sum()
    
    
    if isGPU:
        lstm_pred = torch.max(output_lstm, 1)[1].cuda()
    else:
        lstm_pred = torch.max(output_lstm, 1)[1]
    
    lstm_eval_acc += lstm_eval_correct.item()
    lstm_final_prediction = np.concatenate((lstm_final_prediction, lstm_pred.cpu().numpy()), axis=0)
    lstm_final_test = np.concatenate((lstm_final_test, batch_y), axis=0)

f1 = f1_score(lstm_final_test, lstm_final_prediction, average='binary').item()
recall = recall_score(lstm_final_test, lstm_final_prediction, average='binary').item()
precision = precision_score(lstm_final_test, lstm_final_prediction, average='binary').item()
lstm_eval_acc /= float(len(test_loader.dataset))
lstm_eval_loss /= float(len(test_loader.dataset))
history['test']['acc'].append(lstm_eval_acc)
history['test']['loss'].append(lstm_eval_loss)
history['test']['f1'].append(f1)
history['test']['recall'].append(recall)
history['test']['precision'].append(precision)
print(f'Testing Loss: {lstm_eval_loss}, Accuracy: {lstm_eval_acc}, f1: {f1}, recall: {recall}, precision: {precision}')      



print(history)

with open(f'./history-total.pkl', 'wb') as f:
    pickle.dump(history, f)    

plt.figure()

ep = len(history['train']['loss'])
plt.plot(range(ep), history['train']['loss'], label='LSTM training loss')
plt.plot(range(ep), history['val']['loss'], label='LSTM validation loss')


plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Variation')
plt.savefig('./'+'Loss_LSTM_detection.png')


plt.figure()

ep = len(history['train']['acc'])
plt.plot(range(ep), history['train']['acc'], label='LSTM training acc')
plt.plot(range(ep), history['val']['acc'], label='LSTM validation acc')


plt.legend()

plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.title('Acc Variation')
plt.savefig('./'+'Acc_LSTM_detection.png')



