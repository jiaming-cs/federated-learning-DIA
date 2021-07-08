from model import LSTM
from loader import WaveformDetectionDLPickle
from torch.utils.data import DataLoader
import torch
from torch import nn
import time

import warnings
warnings.filterwarnings('ignore')

DATASET_DIR = './splited_data'
INPUT_SIZE = 6 # num of feature for deep learning
EPOCH = 1
BATCH_SIZE = 128
TIME_STEP = 100  # length of LSTM time sequence, or window size
VAL_SPLIT = 0.1
LR = 0.001   # learning rate
isGPU = torch.cuda.is_available()
loss_func = nn.CrossEntropyLoss()

history = {'train':{'loss':[], 'acc':[]}, 'val':{'loss':[], 'acc':[]}, 'test':{'loss':[], 'acc':[], 'f1':[], 'recall':[], 'precision':[]}}

def load_data(client_num):
    train_dataset = WaveformDetectionDLPickle(DATASET_DIR, client_num)
    test_dataset = WaveformDetectionDLPickle(DATASET_DIR, -1)
    
    return train_dataset, test_dataset


lstm = LSTM()
train_dataset, test_dataset = load_data(0)


if isGPU:
    lstm = nn.DataParallel(lstm, device_ids=[0,1,2,3])
    lstm.cuda()

lstm_optimizer = torch.optim.Adam(lstm.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

training_data, val_data = torch.utils.data.random_split(train_dataset, [int(len(train_dataset) * (1-VAL_SPLIT)), len(train_dataset) - int(len(train_dataset) * (1-VAL_SPLIT))])
training_loader = DataLoader(dataset=training_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)

start = time.time()
print(f"start time:{start}")
for epoch in range(EPOCH):
    lstm_train_loss = 0
    
    lstm_train_acc = 0
    
    lstm.train()
    
    for step, (batch_x, batch_y) in enumerate(training_loader):
        batch_x = batch_x.view(-1, TIME_STEP, INPUT_SIZE)
        print(f"step:{step}")
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
        
end = time.time()
print(f"End time:{end}")
    





